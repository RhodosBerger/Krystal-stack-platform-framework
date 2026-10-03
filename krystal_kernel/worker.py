"""Worker process entry point: `python -m krystal_kernel.worker`.

Protocol (binary, over stdin/stdout): 4-byte big-endian length + pickle frame.
  request  : (req_id:int, kernel:str, payload:dict)
  response : (req_id:int, ok:bool, value_or_error:Any)
Only kernels registered in krystal_kernel.kernels.REGISTRY can run. The worker never imports the
web hub, so startup stays ~50 ms (a spawn-based ProcessPoolExecutor would re-import the server).
"""
import os
import pickle
import struct
import sys


def _read(stream, n):
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def main() -> int:
    from krystal_kernel.kernels import REGISTRY

    rin, rout = sys.stdin.buffer, sys.stdout.buffer
    # Anything the kernels print must not corrupt the protocol channel.
    sys.stdout = sys.stderr
    rout.write(struct.pack(">I", 0))  # ready handshake
    rout.flush()
    while True:
        hdr = _read(rin, 4)
        if hdr is None:
            return 0
        (n,) = struct.unpack(">I", hdr)
        body = _read(rin, n)
        if body is None:
            return 0
        req_id, name, payload = pickle.loads(body)
        try:
            fn = REGISTRY[name]
            resp = (req_id, True, fn(payload))
        except BaseException as e:  # noqa: BLE001 - report everything to the parent
            resp = (req_id, False, f"{type(e).__name__}: {e}")
        data = pickle.dumps(resp, protocol=pickle.HIGHEST_PROTOCOL)
        rout.write(struct.pack(">I", len(data)) + data)
        rout.flush()


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raise SystemExit(main())
