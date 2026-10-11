/**
 * ============================================================================
 * SOVEREIGN ENGINE: GEOMETRIC COMPILER CORE (TypeScript)
 * ============================================================================
 * Compiles declarative JSON/Markdown blueprints into an axial hexagonal
 * topological mesh (q, r), enforces Boolean Hex Ops, strict Pauli Exclusion Guard
 * on layer [A_CENTER], and emits a contiguous 96-byte (24 float32) packed buffer
 * of instances for zero-allocation GPU rendering.
 *
 * Author: Sovereign Engine Architecture Council / Krystal-Stack (2026)
 * ============================================================================
 */

export const VITAL_MAX_HP = 6;
export const MIN_EFFECTIVE_RESISTANCE = 0.1;
export const CONFIDENCE_THRESHOLD = 0.85;

export const LAYER_BASE = 0.0;       // [A_BASE] Hex terrain, elevations
export const LAYER_AURA_RING = 0.1;  // [A_AURA_RING] Selection rings, AoE hazards
export const LAYER_CENTER = 0.5;     // [A_CENTER] Living standees, pawns (Pauli Guarded)
export const LAYER_OVERHEAD = 1.2;   // [A_OVERHEAD] Status runes, conduct badges

export type BiomeType =
  | "PLAINS"
  | "VOLCANIC"
  | "CRYO_FROST"
  | "VOID_CORRUPTION"
  | "CRYSTAL_CITADEL"
  | "CYBER_FORTRESS";

export interface HexCoord {
  q: number;
  r: number;
}

export interface CubeCoord {
  x: number;
  y: number;
  z: number;
}

export interface HexCellData {
  coord: HexCoord;
  biome: BiomeType;
  elevation: number;
  isTraversable: boolean;
  isHazard: boolean;
}

export interface ConductFlags {
  isActive: boolean;
  isLocked: boolean;
  confidence: number;
  isHazardous: boolean;
  vitalHp: number;
  vitalMaxHp: number;
}

export interface EntityInstance {
  id: string;
  name: string;
  coord: HexCoord;
  layer: number; // 0=Base, 1=Aura, 2=Center, 3=Overhead
  hp: number;
  baseResistance: number;
  conduct: ConductFlags;
  colorPbr?: [number, number, number, number]; // r, g, b, roughness
}

export interface BlueprintDefinition {
  mapId: string;
  name: string;
  hexRadius?: number;
  cells: Array<{
    q: number;
    r: number;
    biome?: string;
    elevation?: number;
    is_traversable?: boolean;
    is_hazard?: boolean;
  }>;
  entities: Array<{
    id: string;
    name?: string;
    q: number;
    r: number;
    layer?: string | number;
    hp?: number;
    base_resistance?: number;
    confidence?: number;
    is_hazardous?: boolean;
  }>;
}

export interface CompiledHexMap {
  mapId: string;
  name: string;
  hexRadius: number;
  cells: Map<string, HexCellData>;
  entities: Map<string, EntityInstance>;
  centerOccupancy: Map<string, string>; // CoordKey -> EntityId (Pauli Guard)
  packedBuffer: ArrayBuffer;
  instanceCount: number;
}

// ── 1. AXIAL HEXAGONAL TOPOLOGY MATH ───────────────────────────────────────

export function hexKey(q: number, r: number): string {
  return `${q},${r}`;
}

export function axialToCube(hex: HexCoord): CubeCoord {
  const x = hex.q;
  const z = hex.r;
  const y = -x - z;
  return { x, y, z };
}

export function cubeToAxial(cube: CubeCoord): HexCoord {
  return { q: cube.x, r: cube.z };
}

export function hexDistance(a: HexCoord, b: HexCoord): number {
  const ac = axialToCube(a);
  const bc = axialToCube(b);
  return (Math.abs(ac.x - bc.x) + Math.abs(ac.y - bc.y) + Math.abs(ac.z - bc.z)) / 2;
}

/**
 * Converts axial (q, r) to Pointy-Topped World Coordinates (x, z).
 * x = radius * sqrt(3) * (q + r / 2)
 * z = radius * 1.5 * r
 */
export function axialToWorld(q: number, r: number, radius = 1.0): { x: number; z: number } {
  const x = radius * Math.sqrt(3.0) * (q + r / 2.0);
  const z = radius * 1.5 * r;
  return { x, z };
}

export function worldToAxial(x: number, z: number, radius = 1.0): HexCoord {
  const qFrac = ((Math.sqrt(3) / 3.0) * x - (1.0 / 3.0) * z) / radius;
  const rFrac = ((2.0 / 3.0) * z) / radius;
  return axialRound(qFrac, rFrac);
}

function axialRound(qFrac: number, rFrac: number): HexCoord {
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

// ── 2. CONDUCT CODE BRIDGE & SAFETY BADGES ──────────────────────────────────

export function formatConductBadge(flags: ConductFlags): string {
  const act = flags.isActive ? "[ACT]" : "[!ACT]";
  const lck = flags.isLocked ? "[LCK]" : "[!LCK]";
  const cnf = flags.confidence >= CONFIDENCE_THRESHOLD ? "[CNF]" : "[!CNF]";
  const haz = flags.isHazardous ? "[HAZ]" : "[!HAZ]";
  return `${act} ${lck} ${cnf} ${haz} [VIT ${flags.vitalHp}HP]`;
}

export function validateConductCode(flags: ConductFlags): { valid: boolean; reason?: string } {
  if (!flags.isActive) {
    return { valid: false, reason: "Entity is inactive [!ACT]" };
  }
  if (flags.isHazardous) {
    return { valid: false, reason: "Hazardous operation blocked [HAZ]" };
  }
  if (flags.confidence < CONFIDENCE_THRESHOLD) {
    return { valid: false, reason: `Confidence ${flags.confidence.toFixed(2)} below 0.85 threshold [!CNF]` };
  }
  if (flags.vitalHp <= 0) {
    return { valid: false, reason: "Entity has 0 HP (neutralized)" };
  }
  return { valid: true };
}

// ── 3. PAULI EXCLUSION GUARD ───────────────────────────────────────────────

export class PauliExclusionGuard {
  private occupancy = new Map<string, string>(); // hexKey -> entityId

  public clear(): void {
    this.occupancy.clear();
  }

  public validateAndOccupy(coord: HexCoord, entityId: string, layer: number): { valid: boolean; error?: string } {
    // Strict Pauli Exclusion applies exclusively to layer [A_CENTER] (layer 2)
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

  public vacate(coord: HexCoord, entityId: string): void {
    const key = hexKey(coord.q, coord.r);
    if (this.occupancy.get(key) === entityId) {
      this.occupancy.delete(key);
    }
  }

  public getOccupant(coord: HexCoord): string | undefined {
    return this.occupancy.get(hexKey(coord.q, coord.r));
  }

  public getOccupancyMap(): Map<string, string> {
    return new Map(this.occupancy);
  }
}

// ── 4. BINARY PACKED BUFFER GENERATOR (96 Bytes / Instance) ────────────────

/**
 * Packs instances into contiguous ArrayBuffer of 24 Float32s (96 bytes):
 * - Floats 0..15: 4x4 Transformation Matrix (Column-Major)
 * - Floats 16..19: PBR Color & Roughness (R, G, B, Roughness)
 * - Floats 20..23: Custom Shader Flags & Layer ID & Conduct Flags
 *   - Float 20: Layer ID (0=Base, 1=Aura, 2=Center, 3=Overhead)
 *   - Float 21: Hex Boolean Ops (traversable/hazard bitmask)
 *   - Float 22: Conduct Code bitmask
 *   - Float 23: Entropy factor / Animation pulse
 */
export function generatePackedBuffer(
  cells: Map<string, HexCellData>,
  entities: Map<string, EntityInstance>,
  hexRadius = 1.0
): { buffer: ArrayBuffer; view: Float32Array; totalInstances: number } {
  const totalInstances = cells.size + entities.size;
  const buffer = new ArrayBuffer(totalInstances * 96);
  const view = new Float32Array(buffer);

  let instanceIdx = 0;

  // 1. Pack Terrain Hex Cells (Layer A_BASE = 0.0)
  for (const cell of cells.values()) {
    const offset = instanceIdx * 24;
    const { x, z } = axialToWorld(cell.coord.q, cell.coord.r, hexRadius);
    const y = cell.elevation;

    // 4x4 Matrix (Column-Major)
    view[offset + 0] = 1.0; view[offset + 4] = 0.0; view[offset + 8] = 0.0;  view[offset + 12] = x;
    view[offset + 1] = 0.0; view[offset + 5] = 1.0; view[offset + 9] = 0.0;  view[offset + 13] = y;
    view[offset + 2] = 0.0; view[offset + 6] = 0.0; view[offset + 10] = 1.0; view[offset + 14] = z;
    view[offset + 3] = 0.0; view[offset + 7] = 0.0; view[offset + 11] = 0.0; view[offset + 15] = 1.0;

    // PBR Color based on Biome
    const color = getBiomeColor(cell.biome);
    view[offset + 16] = color[0];
    view[offset + 17] = color[1];
    view[offset + 18] = color[2];
    view[offset + 19] = color[3]; // Roughness

    // Shader Flags
    view[offset + 20] = LAYER_BASE; // Layer ID
    view[offset + 21] = (cell.isTraversable ? 1.0 : 0.0) + (cell.isHazard ? 2.0 : 0.0);
    view[offset + 22] = 7.0; // Terrain default conduct
    view[offset + 23] = 0.05; // Base entropy

    instanceIdx++;
  }

  // 2. Pack Entities (Living Standees, Turrets, Shields)
  for (const entity of entities.values()) {
    const offset = instanceIdx * 24;
    const { x, z } = axialToWorld(entity.coord.q, entity.coord.r, hexRadius);
    const y = entity.layer;

    // 4x4 Matrix
    view[offset + 0] = 1.0; view[offset + 4] = 0.0; view[offset + 8] = 0.0;  view[offset + 12] = x;
    view[offset + 1] = 0.0; view[offset + 5] = 1.0; view[offset + 9] = 0.0;  view[offset + 13] = y;
    view[offset + 2] = 0.0; view[offset + 6] = 0.0; view[offset + 10] = 1.0; view[offset + 14] = z;
    view[offset + 3] = 0.0; view[offset + 7] = 0.0; view[offset + 11] = 0.0; view[offset + 15] = 1.0;

    // Entity Color
    const defaultColor = entity.hp <= 2 ? [0.95, 0.2, 0.1, 0.25] : [0.2, 0.75, 0.95, 0.4];
    const color = entity.colorPbr || defaultColor;
    view[offset + 16] = color[0];
    view[offset + 17] = color[1];
    view[offset + 18] = color[2];
    view[offset + 19] = color[3];

    // Shader Flags
    view[offset + 20] = entity.layer;
    view[offset + 21] = entity.hp; // Encodes HP in flag slot
    let conductMask = 0;
    if (entity.conduct.isActive) conductMask |= 1;
    if (entity.conduct.isLocked) conductMask |= 2;
    if (entity.conduct.confidence >= CONFIDENCE_THRESHOLD) conductMask |= 4;
    if (entity.conduct.isHazardous) conductMask |= 8;
    view[offset + 22] = conductMask;
    view[offset + 23] = 0.15; // Entity pulse factor

    instanceIdx++;
  }

  return { buffer, view, totalInstances };
}

function getBiomeColor(biome: BiomeType): [number, number, number, number] {
  switch (biome) {
    case "CRYSTAL_CITADEL":
      return [0.15, 0.65, 0.95, 0.2];
    case "VOLCANIC":
      return [0.92, 0.25, 0.05, 0.7];
    case "CRYO_FROST":
      return [0.75, 0.9, 1.0, 0.1];
    case "VOID_CORRUPTION":
      return [0.35, 0.05, 0.45, 0.8];
    case "CYBER_FORTRESS":
      return [0.05, 0.85, 0.65, 0.3];
    case "PLAINS":
    default:
      return [0.2, 0.6, 0.3, 0.5];
  }
}

// ── 5. DECLARATIVE BLUEPRINT COMPILER ──────────────────────────────────────

export class GeometricCompiler {
  private pauliGuard = new PauliExclusionGuard();

  public compile(blueprint: BlueprintDefinition): CompiledHexMap {
    this.pauliGuard.clear();
    const cells = new Map<string, HexCellData>();
    const entities = new Map<string, EntityInstance>();
    const hexRadius = blueprint.hexRadius || 1.0;

    // 1. Compile Cells with Boolean Hex Ops
    for (const rawCell of blueprint.cells) {
      const coord: HexCoord = { q: rawCell.q, r: rawCell.r };
      const biome: BiomeType = (rawCell.biome as BiomeType) || "CRYSTAL_CITADEL";
      const isHazard = rawCell.is_hazard ?? (biome === "VOLCANIC" || biome === "VOID_CORRUPTION");
      const isTraversable = rawCell.is_traversable ?? !isHazard;

      const cell: HexCellData = {
        coord,
        biome,
        elevation: rawCell.elevation ?? 0.0,
        isTraversable,
        isHazard
      };
      cells.set(hexKey(coord.q, coord.r), cell);
    }

    // 2. Compile Entities with Pauli Exclusion Guard
    for (const rawEntity of blueprint.entities) {
      const coord: HexCoord = { q: rawEntity.q, r: rawEntity.r };
      let layer = LAYER_CENTER;
      if (typeof rawEntity.layer === "string") {
        if (rawEntity.layer === "A_BASE") layer = LAYER_BASE;
        else if (rawEntity.layer === "A_AURA_RING") layer = LAYER_AURA_RING;
        else if (rawEntity.layer === "A_CENTER") layer = LAYER_CENTER;
        else if (rawEntity.layer === "A_OVERHEAD") layer = LAYER_OVERHEAD;
      } else if (typeof rawEntity.layer === "number") {
        layer = rawEntity.layer;
      }

      // Check Pauli Exclusion Guard
      const guardResult = this.pauliGuard.validateAndOccupy(coord, rawEntity.id, layer);
      if (!guardResult.valid) {
        throw new Error(guardResult.error);
      }

      const hp = Math.max(0, Math.min(VITAL_MAX_HP, rawEntity.hp ?? VITAL_MAX_HP));
      const confidence = rawEntity.confidence ?? 1.0;
      const isHazardous = rawEntity.is_hazardous ?? false;

      const entity: EntityInstance = {
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

    // 3. Generate Binary Packed Buffer
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

  public parseMarkdownBlueprint(markdown: string): BlueprintDefinition {
    // Extracts JSON frontmatter or ```json block from Markdown
    const jsonMatch = markdown.match(/```json([\s\S]*?)```/);
    if (jsonMatch && jsonMatch[1]) {
      return JSON.parse(jsonMatch[1].trim());
    }
    // Fallback: simple line parser
    return {
      mapId: "MD_PARSED_MAP",
      name: "Markdown Sovereign Blueprint",
      cells: [],
      entities: []
    };
  }
}
