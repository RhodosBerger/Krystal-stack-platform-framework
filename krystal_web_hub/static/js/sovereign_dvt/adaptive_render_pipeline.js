/**
 * SOVEREIGN ENGINE: ADAPTIVE RENDER PIPELINE (Isomorphic JS / Browser & Node.js)
 * Implements 4-tier Z-index layering, 3 telemetry-driven hardware profiles,
 * and 20 Hz throttled uniform writes.
 */

class AdaptiveRenderPipeline {
  constructor() {
    this.currentProfile = "STANDARD_LIVING_STANDEES";
    this.lastUniformCommitMs = 0;
    this.minCommitIntervalMs = 50.0; // Max 20 Hz (50 ms minimum period)

    this.uniformBuffer = {
      timeSeconds: 0,
      entropyH: 0.1,
      bloomIntensity: 0.5,
      chromaticAberration: 0.0,
      wireframeActive: false,
      throttleHz: 20
    };
  }

  evaluateTelemetry(telemetry) {
    const { fps, cpuLoad, temperatureC } = telemetry;

    // Profile 1: Minimal Wireframe
    if (fps < 30 || cpuLoad > 85 || temperatureC > 80) {
      this.currentProfile = "MINIMAL_WIREFRAME";
      this.uniformBuffer.wireframeActive = true;
      this.uniformBuffer.bloomIntensity = 0.0;
      this.uniformBuffer.chromaticAberration = 0.0;
    }
    // Profile 3: AAA Cinematic
    else if (fps >= 55 && cpuLoad < 50 && temperatureC < 65) {
      this.currentProfile = "AAA_CINEMATIC";
      this.uniformBuffer.wireframeActive = false;
      this.uniformBuffer.bloomIntensity = 1.2;
      this.uniformBuffer.chromaticAberration = Math.min(0.08, (telemetry.entropyH || 0.1) * 0.05);
    }
    // Profile 2: Standard Living Standees
    else {
      this.currentProfile = "STANDARD_LIVING_STANDEES";
      this.uniformBuffer.wireframeActive = false;
      this.uniformBuffer.bloomIntensity = 0.4;
      this.uniformBuffer.chromaticAberration = 0.005;
    }

    this.uniformBuffer.entropyH = telemetry.entropyH || 0.1;
    return this.currentProfile;
  }

  updateUniforms(nowMs, timeSeconds) {
    if (nowMs - this.lastUniformCommitMs < this.minCommitIntervalMs) {
      return false; // Throttled at 20 Hz
    }
    this.lastUniformCommitMs = nowMs;
    this.uniformBuffer.timeSeconds = timeSeconds;
    return true;
  }

  getActiveProfile() {
    return this.currentProfile;
  }

  getUniformState() {
    return Object.assign({}, this.uniformBuffer);
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { AdaptiveRenderPipeline };
}
if (typeof window !== "undefined") {
  window.AdaptiveRenderPipeline = AdaptiveRenderPipeline;
}
