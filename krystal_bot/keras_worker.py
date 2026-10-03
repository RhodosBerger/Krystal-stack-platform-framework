"""Keras worker. Runs under the *isolated* `.venv-keras` interpreter, never in the hub process.

    .venv-keras\\Scripts\\python.exe -m krystal_bot.keras_worker

Protocol: one JSON object per line on stdin -> one JSON object per line on stdout.
  {"op": "ping"}
  {"op": "train", "x": [[...]], "y": [0|1,...], "model_path": "...", "epochs": 60, "seed": 7}
  {"op": "predict", "x": [[...]], "model_path": "..."}
Only numeric arrays and a path are accepted; nothing is evaluated. The hub keeps working (with the
stdlib fallback model) when this worker or Keras is unavailable.
"""
import json
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def _auc(y, p):
    pos = [(s, i) for i, (s, t) in enumerate(zip(p, y)) if t == 1]
    neg = [s for s, t in zip(p, y) if t == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for s, _ in pos:
        for n in neg:
            wins += 1.0 if s > n else 0.5 if s == n else 0.0
    return wins / (len(pos) * len(neg))


def main() -> int:
    out = sys.stdout
    sys.stdout = sys.stderr  # keep library chatter off the protocol channel
    import numpy as np
    import keras
    import torch

    torch.set_num_threads(max(1, int(os.environ.get("KRYSTAL_KERAS_THREADS", "2"))))  # leave cores for the hub
    models = {}

    def reply(obj):
        out.write(json.dumps(obj) + "\n")
        out.flush()

    reply({"ok": True, "ready": True, "keras": keras.__version__, "backend": keras.backend.backend(), "torch": torch.__version__})
    for line in sys.stdin:
        try:
            req = json.loads(line)
            op = req.get("op")
            if op == "ping":
                reply({"ok": True, "keras": keras.__version__, "backend": keras.backend.backend()})
            elif op == "train":
                x = np.asarray(req["x"], dtype="float32")
                y = np.asarray(req["y"], dtype="float32")
                if x.ndim != 2 or len(x) != len(y) or len(x) < 20:
                    raise ValueError("need a 2-D matrix with >= 20 rows and matching labels")
                keras.utils.set_random_seed(int(req.get("seed", 7)))
                rng = np.random.default_rng(int(req.get("seed", 7)))
                idx = rng.permutation(len(x))
                cut = max(10, int(0.8 * len(x)))
                tr, ho = idx[:cut], idx[cut:]
                mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
                norm = keras.layers.Normalization(axis=-1)
                norm.adapt(x[tr])
                model = keras.Sequential([
                    keras.layers.Input((x.shape[1],)), norm,
                    keras.layers.Dense(16, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-3)),
                    keras.layers.Dense(8, activation="relu"),
                    keras.layers.Dense(1, activation="sigmoid")])
                model.compile(optimizer=keras.optimizers.Adam(0.01), loss="binary_crossentropy")
                npos = max(1.0, float(y[tr].sum()))
                nneg = max(1.0, float(len(tr) - y[tr].sum()))
                cw = {0: len(tr) / (2 * nneg), 1: len(tr) / (2 * npos)}  # counter class imbalance
                hist = model.fit(x[tr], y[tr], epochs=int(req.get("epochs", 60)), batch_size=32, verbose=0, class_weight=cw,
                                 validation_data=(x[ho], y[ho]) if len(ho) else None,
                                 callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)] if len(ho) else None)
                p = model.predict(x[ho], verbose=0).ravel().tolist() if len(ho) else []
                path = req["model_path"]
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                model.save(path)
                models[path] = model
                reply({"ok": True, "epochs_run": len(hist.history["loss"]), "loss": float(hist.history["loss"][-1]),
                       "holdout_auc": _auc(y[ho].tolist(), p), "holdout_n": int(len(ho)),
                       "holdout_pos": int(y[ho].sum()), "params": int(model.count_params())})
            elif op == "predict":
                path = req["model_path"]
                if path not in models:
                    models[path] = keras.saving.load_model(path)
                x = np.asarray(req["x"], dtype="float32")
                reply({"ok": True, "p": models[path].predict(x, verbose=0).ravel().tolist()})
            else:
                reply({"ok": False, "error": f"unknown op {op!r}"})
        except BaseException as e:  # noqa: BLE001
            reply({"ok": False, "error": f"{type(e).__name__}: {e}"})
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raise SystemExit(main())
