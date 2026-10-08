/**
 * KRYSTAL-STACK NEXTGEN: DEDICATED MULTI-THREADED WEB WORKER
 * ==========================================================
 * Offloads frame decoding, ASCII string buffer normalization,
 * tokenization, and metric aggregation from the main browser event loop.
 *
 * Invariant: VITAL_MAX_HP = 6
 */

const VITAL_MAX_HP = 6;

// Interned symbol dictionary in Worker scope for fast token matching
const SYMBOL_CACHE = new Map();
let symbolCounter = 1;

function intern(str) {
  if (SYMBOL_CACHE.has(str)) return SYMBOL_CACHE.get(str);
  const id = symbolCounter++;
  SYMBOL_CACHE.set(str, id);
  return id;
}

self.onmessage = function (e) {
  const msg = e.data;
  if (!msg || !msg.type) return;

  switch (msg.type) {
    case "INIT":
      self.postMessage({
        type: "INITIALIZED",
        workerId: "KRYSTAL_WORKER_01",
        vitalHp: VITAL_MAX_HP,
        timestamp: performance.now()
      });
      break;

    case "PROCESS_FRAME": {
      // Decode and process frame off the main UI thread
      const payload = msg.payload;
      const t0 = performance.now();

      let processedAscii = payload.ascii || "";
      // Strip carriage returns and ensure uniform line endings
      if (processedAscii.includes("\r")) {
        processedAscii = processedAscii.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
      }

      // Compute visual entropy metrics off-thread
      const len = processedAscii.length;
      let nonSpaceCount = 0;
      for (let i = 0; i < len; i++) {
        const code = processedAscii.charCodeAt(i);
        if (code > 32) nonSpaceCount++;
      }
      const density = len > 0 ? nonSpaceCount / len : 0.0;
      const dt = performance.now() - t0;

      self.postMessage({
        type: "FRAME_PROCESSED",
        frameId: payload.frame_id || 0,
        ascii: processedAscii,
        fps: payload.fps || 60.0,
        mode: payload.mode || "CYBERPUNK",
        entropy: payload.entropy || density,
        density: density,
        workerDurationMs: dt,
        vitalHp: VITAL_MAX_HP,
        governor: payload.governor,
        backpressure: payload.backpressure,
        cognitive_phase: payload.cognitive_phase,
        hamiltonian_energy: payload.hamiltonian_energy,
        vulkan: payload.vulkan
      });
      break;
    }

    case "FAST_TOKENIZE": {
      const text = msg.text || "";
      const t0 = performance.now();
      const tokens = text.split(/\s+/).filter(Boolean);
      const tokenIds = new Int32Array(tokens.length);
      for (let i = 0; i < tokens.length; i++) {
        tokenIds[i] = intern(tokens[i]);
      }
      const dt = performance.now() - t0;

      self.postMessage({
        type: "TOKENIZE_DONE",
        tokenCount: tokens.length,
        tokenIds: Array.from(tokenIds),
        durationMs: dt,
        vitalHp: VITAL_MAX_HP
      });
      break;
    }

    case "BENCHMARK_JS": {
      // Benchmark JS engine multi-threaded matrix throughput in worker
      const size = msg.size || 128;
      const t0 = performance.now();
      const a = new Float32Array(size * size).fill(1.001);
      const b = new Float32Array(size * size).fill(1.002);
      const c = new Float32Array(size * size);

      for (let i = 0; i < size; i++) {
        for (let k = 0; k < size; k++) {
          const aik = a[i * size + k];
          for (let j = 0; j < size; j++) {
            c[i * size + j] += aik * b[k * size + j];
          }
        }
      }
      const dt = performance.now() - t0;
      const gflops = (2 * (size ** 3)) / (dt * 1e6);

      self.postMessage({
        type: "BENCHMARK_DONE",
        size: size,
        durationMs: dt,
        gflops: gflops,
        vitalHp: VITAL_MAX_HP
      });
      break;
    }

    default:
      console.warn("[Worker] Unknown message type:", msg.type);
  }
};
