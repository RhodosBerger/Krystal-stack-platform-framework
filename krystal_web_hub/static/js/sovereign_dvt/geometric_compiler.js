/**
 * SOVEREIGN ENGINE: GEOMETRIC COMPILER (Isomorphic JS / Browser & Node.js)
 * Implements axial hex topology (q, r), Boolean Hex Ops, Pauli Exclusion Guard on [A_CENTER],
 * and 96-byte packed instance buffer generation (24 float32s).
 */

const VITAL_MAX_HP = 6;
const MIN_EFFECTIVE_RESISTANCE = 0.1;
const CONFIDENCE_THRESHOLD = 0.85;

const LAYER_BASE = 0.0;       // [A_BASE]
const LAYER_AURA_RING = 0.1;  // [A_AURA_RING]
const LAYER_CENTER = 0.5;     // [A_CENTER] - Pauli Exclusion Guarded
const LAYER_OVERHEAD = 1.2;   // [A_OVERHEAD]

function hexKey(q, r) {
  return `${q},${r}`;
}

function axialToCube(hex) {
  const x = hex.q;
  const z = hex.r;
  const y = -x - z;
  return { x, y, z };
}

function cubeToAxial(cube) {
  return { q: cube.x, r: cube.z };
}

function hexDistance(a, b) {
  const ac = axialToCube(a);
  const bc = axialToCube(b);
  return (Math.abs(ac.x - bc.x) + Math.abs(ac.y - bc.y) + Math.abs(ac.z - bc.z)) / 2;
}

function axialToWorld(q, r, radius = 1.0) {
  const x = radius * Math.sqrt(3.0) * (q + r / 2.0);
  const z = radius * 1.5 * r;
  return { x, z };
}

function worldToAxial(x, z, radius = 1.0) {
  const qFrac = ((Math.sqrt(3) / 3.0) * x - (1.0 / 3.0) * z) / radius;
  const rFrac = ((2.0 / 3.0) * z) / radius;
  return axialRound(qFrac, rFrac);
}

function axialRound(qFrac, rFrac) {
  const x = qFrac;
  const z = rFrac;
  const y = -x - z;

  let rx = Math.round(x);
  let ry = Math.round(y);
  let rz = Math.round(z);

  const xDiff = Math.abs(rx - x);
  const yDiff = Math.abs(ry - y);
  const zDiff = Math.abs(rz - z);

  if (xDiff > yDiff && xDiff > zDiff) {
    rx = -ry - rz;
  } else if (yDiff > zDiff) {
    ry = -rx - rz;
  } else {
    rz = -rx - ry;
  }
  return { q: rx, r: rz };
}

function formatConductBadge(flags) {
  const act = flags.isActive ? "[ACT]" : "[!ACT]";
  const lck = flags.isLocked ? "[LCK]" : "[!LCK]";
  const cnf = flags.confidence >= CONFIDENCE_THRESHOLD ? "[CNF]" : "[!CNF]";
  const haz = flags.isHazardous ? "[HAZ]" : "[!HAZ]";
  return `${act} ${lck} ${cnf} ${haz} [VIT ${flags.vitalHp}HP]`;
}

function validateConductCode(flags) {
  if (!flags.isActive) return { valid: false, reason: "Entity is inactive [!ACT]" };
  if (flags.isHazardous) return { valid: false, reason: "Hazardous operation blocked [HAZ]" };
  if (flags.confidence < CONFIDENCE_THRESHOLD) {
    return { valid: false, reason: `Confidence ${flags.confidence.toFixed(2)} below 0.85 threshold [!CNF]` };
  }
  if (flags.vitalHp <= 0) return { valid: false, reason: "Entity has 0 HP (neutralized)" };
  return { valid: true };
}

class PauliExclusionGuard {
  constructor() {
    this.occupancy = new Map();
  }

  clear() {
    this.occupancy.clear();
  }

  validateAndOccupy(coord, entityId, layer) {
    if (Math.abs(layer - LAYER_CENTER) > 0.05 && layer !== 2) {
      return { valid: true };
    }
    const key = hexKey(coord.q, coord.r);
    const existing = this.occupancy.get(key);
    if (existing && existing !== entityId) {
      return {
        valid: false,
        error: `Pauli Exclusion Violation: Hex (${coord.q}, ${coord.r}) on [A_CENTER] already occupied by ${existing}`
      };
    }
    this.occupancy.set(key, entityId);
    return { valid: true };
  }

  vacate(coord, entityId) {
    const key = hexKey(coord.q, coord.r);
    if (this.occupancy.get(key) === entityId) {
      this.occupancy.delete(key);
    }
  }

  getOccupant(coord) {
    return this.occupancy.get(hexKey(coord.q, coord.r));
  }

  getOccupancyMap() {
    return new Map(this.occupancy);
  }
}

function getBiomeColor(biome) {
  switch (biome) {
    case "CRYSTAL_CITADEL": return [0.15, 0.65, 0.95, 0.2];
    case "VOLCANIC":        return [0.92, 0.25, 0.05, 0.7];
    case "CRYO_FROST":       return [0.75, 0.90, 1.00, 0.1];
    case "VOID_CORRUPTION": return [0.35, 0.05, 0.45, 0.8];
    case "CYBER_FORTRESS":  return [0.05, 0.85, 0.65, 0.3];
    case "PLAINS":
    default:                return [0.20, 0.60, 0.30, 0.5];
  }
}

function generatePackedBuffer(cells, entities, hexRadius = 1.0) {
  const totalInstances = cells.size + entities.size;
  const buffer = new ArrayBuffer(totalInstances * 96);
  const view = new Float32Array(buffer);

  let instanceIdx = 0;

  // 1. Pack Cells (A_BASE)
  for (const cell of cells.values()) {
    const offset = instanceIdx * 24;
    const { x, z } = axialToWorld(cell.coord.q, cell.coord.r, hexRadius);
    const y = cell.elevation;

    // 4x4 matrix (column-major)
    view[offset + 0] = 1.0; view[offset + 4] = 0.0; view[offset + 8] = 0.0;  view[offset + 12] = x;
    view[offset + 1] = 0.0; view[offset + 5] = 1.0; view[offset + 9] = 0.0;  view[offset + 13] = y;
    view[offset + 2] = 0.0; view[offset + 6] = 0.0; view[offset + 10] = 1.0; view[offset + 14] = z;
    view[offset + 3] = 0.0; view[offset + 7] = 0.0; view[offset + 11] = 0.0; view[offset + 15] = 1.0;

    const color = getBiomeColor(cell.biome);
    view[offset + 16] = color[0];
    view[offset + 17] = color[1];
    view[offset + 18] = color[2];
    view[offset + 19] = color[3];

    view[offset + 20] = LAYER_BASE;
    view[offset + 21] = (cell.isTraversable ? 1.0 : 0.0) + (cell.isHazard ? 2.0 : 0.0);
    view[offset + 22] = 7.0; // default conduct
    view[offset + 23] = 0.05;

    instanceIdx++;
  }

  // 2. Pack Entities
  for (const entity of entities.values()) {
    const offset = instanceIdx * 24;
    const { x, z } = axialToWorld(entity.coord.q, entity.coord.r, hexRadius);
    const y = entity.layer;

    view[offset + 0] = 1.0; view[offset + 4] = 0.0; view[offset + 8] = 0.0;  view[offset + 12] = x;
    view[offset + 1] = 0.0; view[offset + 5] = 1.0; view[offset + 9] = 0.0;  view[offset + 13] = y;
    view[offset + 2] = 0.0; view[offset + 6] = 0.0; view[offset + 10] = 1.0; view[offset + 14] = z;
    view[offset + 3] = 0.0; view[offset + 7] = 0.0; view[offset + 11] = 0.0; view[offset + 15] = 1.0;

    const defaultColor = entity.hp <= 2 ? [0.95, 0.2, 0.1, 0.25] : [0.2, 0.75, 0.95, 0.4];
    const color = entity.colorPbr || defaultColor;
    view[offset + 16] = color[0];
    view[offset + 17] = color[1];
    view[offset + 18] = color[2];
    view[offset + 19] = color[3];

    view[offset + 20] = entity.layer;
    view[offset + 21] = entity.hp;
    let conductMask = 0;
    if (entity.conduct.isActive) conductMask |= 1;
    if (entity.conduct.isLocked) conductMask |= 2;
    if (entity.conduct.confidence >= CONFIDENCE_THRESHOLD) conductMask |= 4;
    if (entity.conduct.isHazardous) conductMask |= 8;
    view[offset + 22] = conductMask;
    view[offset + 23] = 0.15;

    instanceIdx++;
  }

  return { buffer, view, totalInstances };
}

class GeometricCompiler {
  constructor() {
    this.pauliGuard = new PauliExclusionGuard();
  }

  compile(blueprint) {
    this.pauliGuard.clear();
    const cells = new Map();
    const entities = new Map();
    const hexRadius = blueprint.hexRadius || 1.0;

    for (const rawCell of blueprint.cells) {
      const coord = { q: rawCell.q, r: rawCell.r };
      const biome = rawCell.biome || "CRYSTAL_CITADEL";
      const isHazard = rawCell.is_hazard ?? (biome === "VOLCANIC" || biome === "VOID_CORRUPTION");
      const isTraversable = rawCell.is_traversable ?? !isHazard;

      const cell = {
        coord,
        biome,
        elevation: rawCell.elevation ?? 0.0,
        isTraversable,
        isHazard
      };
      cells.set(hexKey(coord.q, coord.r), cell);
    }

    for (const rawEntity of blueprint.entities) {
      const coord = { q: rawEntity.q, r: rawEntity.r };
      let layer = LAYER_CENTER;
      if (typeof rawEntity.layer === "string") {
        if (rawEntity.layer === "A_BASE") layer = LAYER_BASE;
        else if (rawEntity.layer === "A_AURA_RING") layer = LAYER_AURA_RING;
        else if (rawEntity.layer === "A_CENTER") layer = LAYER_CENTER;
        else if (rawEntity.layer === "A_OVERHEAD") layer = LAYER_OVERHEAD;
      } else if (typeof rawEntity.layer === "number") {
        layer = rawEntity.layer;
      }

      const guardResult = this.pauliGuard.validateAndOccupy(coord, rawEntity.id, layer);
      if (!guardResult.valid) {
        throw new Error(guardResult.error);
      }

      const hp = Math.max(0, Math.min(VITAL_MAX_HP, rawEntity.hp ?? VITAL_MAX_HP));
      const confidence = rawEntity.confidence ?? 1.0;
      const isHazardous = rawEntity.is_hazardous ?? false;

      const entity = {
        id: rawEntity.id,
        name: rawEntity.name || rawEntity.id,
        coord,
        layer,
        hp,
        baseResistance: rawEntity.base_resistance ?? 1.5,
        conduct: {
          isActive: hp > 0,
          isLocked: false,
          confidence,
          isHazardous,
          vitalHp: hp,
          vitalMaxHp: VITAL_MAX_HP
        }
      };
      entities.set(entity.id, entity);
    }

    const { buffer, totalInstances } = generatePackedBuffer(cells, entities, hexRadius);

    return {
      mapId: blueprint.mapId,
      name: blueprint.name,
      hexRadius,
      cells,
      entities,
      centerOccupancy: this.pauliGuard.getOccupancyMap(),
      packedBuffer: buffer,
      instanceCount: totalInstances
    };
  }

  parseMarkdownBlueprint(markdown) {
    const jsonMatch = markdown.match(/```json([\s\S]*?)```/);
    if (jsonMatch && jsonMatch[1]) {
      return JSON.parse(jsonMatch[1].trim());
    }
    return {
      mapId: "MD_PARSED_MAP",
      name: "Markdown Sovereign Blueprint",
      cells: [],
      entities: []
    };
  }
}

// Module export for Node & Browser
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    VITAL_MAX_HP,
    MIN_EFFECTIVE_RESISTANCE,
    CONFIDENCE_THRESHOLD,
    LAYER_BASE,
    LAYER_AURA_RING,
    LAYER_CENTER,
    LAYER_OVERHEAD,
    hexKey,
    axialToCube,
    cubeToAxial,
    hexDistance,
    axialToWorld,
    worldToAxial,
    formatConductBadge,
    validateConductCode,
    PauliExclusionGuard,
    generatePackedBuffer,
    GeometricCompiler
  };
}
if (typeof window !== "undefined") {
  window.SovereignGeometric = {
    VITAL_MAX_HP,
    MIN_EFFECTIVE_RESISTANCE,
    CONFIDENCE_THRESHOLD,
    LAYER_BASE,
    LAYER_AURA_RING,
    LAYER_CENTER,
    LAYER_OVERHEAD,
    hexKey,
    axialToCube,
    cubeToAxial,
    hexDistance,
    axialToWorld,
    worldToAxial,
    formatConductBadge,
    validateConductCode,
    PauliExclusionGuard,
    generatePackedBuffer,
    GeometricCompiler
  };
}
