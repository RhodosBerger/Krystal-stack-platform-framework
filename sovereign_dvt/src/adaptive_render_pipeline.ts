/**
 * ============================================================================
 * SOVEREIGN ENGINE: ADAPTIVE RENDER PIPELINE & SHADER DISPATCHER (TypeScript)
 * ============================================================================
 * 4-Tier Z-Index Layering:
 *   [A_BASE]     (0.0) -> Terrain Hex Mesh
 *   [A_AURA_RING](0.1) -> Selection rings, AOE hazard zones
 *   [A_CENTER]   (0.5) -> Living Standees / 3D pawns (Pauli Guarded)
 *   [A_OVERHEAD] (1.2) -> Floating status runes, conduct badges
 *
 * 3 Adaptive Hardware Profiles governed by live telemetry:
 *   1. Minimal Wireframe (FPS < 30 || CPU > 85% || Temp > 80°C)
 *   2. Standard Living Standees (30 <= FPS < 55 && Temp <= 75°C)
 *   3. AAA Cinematic (FPS >= 55 && CPU < 50% && Temp < 65°C)
 *
 * Zero-allocation rendering with throttled uniform writes (max 20 Hz).
 *
 * Author: Sovereign Engine Architecture Council / Krystal-Stack (2026)
 * ============================================================================
 */

import { LAYER_BASE, LAYER_AURA_RING, LAYER_CENTER, LAYER_OVERHEAD } from "./geometric_compiler";

export type HardwareProfileType =
  | "MINIMAL_WIREFRAME"
  | "STANDARD_LIVING_STANDEES"
  | "AAA_CINEMATIC";

export interface RenderTelemetry {
  fps: number;
  cpuLoad: number;      // 0..100%
  temperatureC: number; // Spindle / CPU Temp °C
  entropyH: number;     // Shannon entropy 0..1
}

export interface UniformState {
  timeSeconds: number;
  entropyH: number;
  bloomIntensity: number;
  chromaticAberration: number;
  wireframeActive: boolean;
  throttleHz: number;
}

export class AdaptiveRenderPipeline {
  private currentProfile: HardwareProfileType = "STANDARD_LIVING_STANDEES";
  private lastUniformCommitMs = 0;
  private readonly minCommitIntervalMs = 50.0; // Max 20 Hz update frequency

  // Preallocated uniform cache (avoids object allocations per frame)
  private uniformBuffer: UniformState = {
    timeSeconds: 0,
    entropyH: 0.1,
    bloomIntensity: 0.5,
    chromaticAberration: 0.0,
    wireframeActive: false,
    throttleHz: 20
  };

  /**
   * Evaluates telemetry and switches hardware profile dynamically.
   */
  public evaluateTelemetry(telemetry: RenderTelemetry): HardwareProfileType {
    const { fps, cpuLoad, temperatureC } = telemetry;

    // Profile 1: Minimal Wireframe (Critical thermal or low framerate state)
    if (fps < 30 || cpuLoad > 85 || temperatureC > 80) {
      this.currentProfile = "MINIMAL_WIREFRAME";
      this.uniformBuffer.wireframeActive = true;
      this.uniformBuffer.bloomIntensity = 0.0;
      this.uniformBuffer.chromaticAberration = 0.0;
    }
    // Profile 3: AAA Cinematic (High headroom state)
    else if (fps >= 55 && cpuLoad < 50 && temperatureC < 65) {
      this.currentProfile = "AAA_CINEMATIC";
      this.uniformBuffer.wireframeActive = false;
      this.uniformBuffer.bloomIntensity = 1.2;
      // Chromatic aberration tied directly to system entropy H
      this.uniformBuffer.chromaticAberration = Math.min(0.08, telemetry.entropyH * 0.05);
    }
    // Profile 2: Standard Living Standees (Balanced state)
    else {
      this.currentProfile = "STANDARD_LIVING_STANDEES";
      this.uniformBuffer.wireframeActive = false;
      this.uniformBuffer.bloomIntensity = 0.4;
      this.uniformBuffer.chromaticAberration = 0.005;
    }

    this.uniformBuffer.entropyH = telemetry.entropyH;
    return this.currentProfile;
  }

  /**
   * Throttled uniform update committing shader parameters at maximum 20 Hz.
   * Returns true if uniform commit was dispatched, false if throttled.
   */
  public updateUniforms(nowMs: number, timeSeconds: number): boolean {
    if (nowMs - this.lastUniformCommitMs < this.minCommitIntervalMs) {
      return false; // Throttled (skips bus write to keep render loop fast)
    }

    this.lastUniformCommitMs = nowMs;
    this.uniformBuffer.timeSeconds = timeSeconds;
    return true;
  }

  public getActiveProfile(): HardwareProfileType {
    return this.currentProfile;
  }

  public getUniformState(): Readonly<UniformState> {
    return this.uniformBuffer;
  }

  public getLayerZIndex(layerName: "A_BASE" | "A_AURA_RING" | "A_CENTER" | "A_OVERHEAD"): number {
    switch (layerName) {
      case "A_BASE": return LAYER_BASE;
      case "A_AURA_RING": return LAYER_AURA_RING;
      case "A_CENTER": return LAYER_CENTER;
      case "A_OVERHEAD": return LAYER_OVERHEAD;
    }
  }
}
