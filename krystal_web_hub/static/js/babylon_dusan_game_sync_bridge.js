/**
 * KRYSTAL-STACK × DUSAN-GAME // BABYLON.JS 3D HEX-GRID & TENSOR SYNCHRONIZATION BRIDGE
 * ====================================================================================
 * Unifies:
 *   1. Babylon.js WebGPU / WebGL2 Scene Graph (64-Hex Arena, ThreadMiniatures, Caustic Shaders)
 *   2. `dusan-game` `ConnectionEngine` (`initial_map`, `state_resolved`, `instruction_compiling`)
 *   3. `AlgebraicActionTensor` (T_{i,j,k}: Ohm's Law I = V / R <= 8.0 A + CV 6D Optical Vector)
 *   4. 24-Byte `KMB1` Mechanical Grid Binary & 128-Byte `ZeroCopyContractFrame` (`VITAL_MAX_HP == 6`)
 */

export class BabylonDusanGameSyncBridge {
  constructor({
    krystalApiUrl = "http://127.0.0.1:8090",
    microServerUrl = "http://127.0.0.1:8091",
    canvasElementId = "krystal-babylon-canvas",
  } = {}) {
    this.krystalApiUrl = krystalApiUrl.replace(/\/$/, "");
    this.microServerUrl = microServerUrl.replace(/\/$/, "");
    this.canvasElementId = canvasElementId;
    this.vitalMaxHp = 6;
    this.criticalCurrentThresholdAmps = 8.0;

    // Pre-allocated Zero-GC ArrayBuffers (128B Constitutional Frame + 24B KMB1 Grid Packet)
    this.ringBuffer128B = new ArrayBuffer(128);
    this.ringView128B = new DataView(this.ringBuffer128B);
    this.kmb1Buffer24B = new ArrayBuffer(24);
    this.kmb1View24B = new DataView(this.kmb1Buffer24B);

    this.hexMeshes = new Map();
    this.threadMeshes = new Map();
    this.connectionState = "DISCONNECTED";
  }

  /**
   * Initializes Babylon.js Scene with Flat-Design HUD Tokens & Hexagonal Grid Instances.
   */
  initBabylonScene(BABYLON) {
    const canvas = document.getElementById(this.canvasElementId);
    if (!canvas || !BABYLON) {
      this.connectionState = "SANDBOX_HEADLESS";
      return { status: "SANDBOX_HEADLESS", vitalMaxHp: this.vitalMaxHp };
    }

    this.engine = new BABYLON.Engine(canvas, true, { preserveDrawingBuffer: true, stencil: true });
    this.scene = new BABYLON.Scene(this.engine);
    this.scene.clearColor = new BABYLON.Color4(0.055, 0.067, 0.086, 1.0); // #0E1116 Obsidian Workshop

    this.camera = new BABYLON.ArcRotateCamera(
      "KrystalIsoCam",
      -Math.PI / 2,
      Math.PI / 3.2,
      26,
      BABYLON.Vector3.Zero(),
      this.scene
    );
    this.camera.attachControl(canvas, true);

    const hemiLight = new BABYLON.HemisphericLight("VrayGI", new BABYLON.Vector3(0, 1, 0), this.scene);
    hemiLight.intensity = 0.72;

    this.buildHexagonalGridMeshes(BABYLON, 3);
    this.engine.runRenderLoop(() => {
      this.scene.render();
    });

    this.connectionState = "CONNECTED";
    return { status: "BABYLON_SCENE_ACTIVE", hexCount: this.hexMeshes.size, vitalMaxHp: this.vitalMaxHp };
  }

  buildHexagonalGridMeshes(BABYLON, radius = 3) {
    const hexRadius = 1.15;
    for (let q = -radius; q <= radius; q++) {
      const r1 = Math.max(-radius, -q - radius);
      const r2 = Math.min(radius, -q + radius);
      for (let r = r1; r <= r2; r++) {
        const key = `${q},${r}`;
        const x = hexRadius * Math.sqrt(3) * (q + r / 2);
        const z = hexRadius * (3 / 2) * r;
        const disc = BABYLON.MeshBuilder.CreateCylinder(
          `hex_${key}`,
          { diameter: hexRadius * 1.88, height: 0.28, tessellation: 6 },
          this.scene
        );
        disc.position.set(x, 0, z);
        this.hexMeshes.set(key, disc);
      }
    }
  }

  /**
   * Packs an 8x8 mechanical grid movement into the exact 24-byte KMB1 binary frame
   * shared across Babylon.js, Godot 4.3, PostgreSQL JSONB, and WordPress Oxygen ACF.
   */
  packKmb1BinaryFrame({
    bitboardLow32 = 0x00000080,
    bitboardHigh32 = 0x81000008,
    fromIdx = 0,
    toIdx = 19,
    lensCode = 1,
    mariasScore = 46,
    crc32 = 0x6ff47f72,
  } = {}) {
    if (this.vitalMaxHp !== 6) {
      throw new Error(`ContractViolation: VITAL_MAX_HP=${this.vitalMaxHp} != 6`);
    }
    const v = this.kmb1View24B;
    // 'KMB1' magic (0x4B4D4231)
    v.setUint8(0, 0x4b);
    v.setUint8(1, 0x4d);
    v.setUint8(2, 0x42);
    v.setUint8(3, 0x31);
    v.setUint32(4, bitboardLow32 >>> 0, true);
    v.setUint32(8, bitboardHigh32 >>> 0, true);
    v.setUint16(12, ((fromIdx & 0xff) << 8) | (toIdx & 0xff), true);
    v.setUint16(14, lensCode & 0xffff, true);
    v.setUint16(16, mariasScore & 0xffff, true);
    v.setUint16(18, this.vitalMaxHp, true);
    v.setUint32(20, crc32 >>> 0, true);
    return new Uint8Array(this.kmb1Buffer24B);
  }

  /**
   * Synchronizes state with the unified Krystal + dusan-game Vulkan Power & Tensor endpoint.
   */
  async syncUnifiedTensorAndPowerState(umaUtilization = 0.35) {
    const res = await fetch(
      `${this.krystalApiUrl}/api/v1/runtime/dusan_game_vulkan_power_plan?uma=${encodeURIComponent(umaUtilization)}`
    );
    if (!res.ok) {
      throw new Error(`Sync failed with HTTP ${res.status}`);
    }
    return await res.json();
  }
}
