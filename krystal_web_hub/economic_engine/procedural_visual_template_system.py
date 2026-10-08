"""
procedural_visual_template_system.py
=====================================
Krystal-Stack Platform Framework: Procedural Visual Template System v1.0.
Implements the 10-block procedural visual specification, temporal narrative layers,
texture stacks, and prompt synthesizers for high-fidelity procedural graphics.

Key Pillars:
1. Nádherná procedurálna grafika (Signed Distance Fields, fractals, Voronoi, dihedral folds).
2. Silná práca s textúrami (Layered Texture Stack: base, secondary, micro, edges, emission).
3. Scéna, ktorá znázorňuje dej (Primary event, cause, visible transformation, environmental response).
4. Vizuálna logika sveta & Temporal Scene Layer (Before, During, After).
5. Multiformat Export (Master prompt, Narrative prompt, Ultra-compact prompt, Godot GLSL Uniforms, JSON, YAML).

Invariant: VITAL_MAX_HP = 6
"""

from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional

VITAL_MAX_HP: int = 6
GOLDEN_RATIO: float = 1.61803398875


@dataclass
class SceneIdentity:
    scene_type: str
    core_theme: str
    narrative_role: str


@dataclass
class WorldEnvironment:
    environment: str
    scale: str
    terrain_logic: str


@dataclass
class ProceduralStructure:
    procedural_forms: List[str]
    fractal_dimension: float = 2.45
    dihedral_folds: int = 6
    octaves: int = 6


@dataclass
class TextureStack:
    base_material: str
    secondary_material: str
    micro_surface_detail: str
    edge_behavior: str
    reflectivity_roughness: str
    emission: str
    atmospheric_particles: str


@dataclass
class ActionEvent:
    primary_event: str
    cause: str
    visible_transformation: str
    environmental_response: str
    motion_cues: str
    narrative_traces: str


@dataclass
class TemporalLayer:
    before: str
    during: str
    after: str


@dataclass
class AtmosphereBlock:
    atmosphere: str
    mood: str
    weather_medium: str
    depth_effects: str


@dataclass
class LightingBlock:
    primary_light: str
    secondary_light: str
    accent_light: str
    shadow_behavior: str


@dataclass
class CameraComposition:
    camera_angle: str
    framing: str
    primary_focal_point: str
    secondary_focal_points: str
    scale_references: str
    composition: str


@dataclass
class StyleRenderMode:
    render_mode: str
    surface_style: str


@dataclass
class QualityDetail:
    detail_level: str = "very high"
    texture_richness: str = "layered and premium"
    world_coherence: str = "physically believable but mythic"
    visual_density: str = "rich focal detail, controlled peripheral softness"


@dataclass
class ProceduralVisualTemplate:
    template_id: str
    title: str
    scene_identity: SceneIdentity
    world_environment: WorldEnvironment
    procedural_structure: ProceduralStructure
    texture_stack: TextureStack
    action_event: ActionEvent
    temporal_layer: TemporalLayer
    atmosphere: AtmosphereBlock
    lighting: LightingBlock
    camera_composition: CameraComposition
    style_render_mode: StyleRenderMode
    quality_detail: QualityDetail = field(default_factory=QualityDetail)
    negative_constraints: List[str] = field(default_factory=lambda: [
        "generic game UI",
        "flat empty surfaces",
        "random unmotivated objects",
        "oversaturated cartoon look",
        "cheap fantasy cliché",
        "blurry undefined texture work",
        "chaotic unreadable composition",
        "too many equal focal points"
    ])
    schema_version: str = "1.0"
    vital_max_hp_rule: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp_rule > VITAL_MAX_HP:
            self.vital_max_hp_rule = VITAL_MAX_HP

    # ── SYNTHESIS: MASTER PROMPT TEMPLATE ──────────────────────────────────
    def to_master_prompt(self) -> str:
        """
        Synthesizes the complete Master Prompt following the official
        Krystal Procedural Visual Template System v1.0 standard.
        """
        si = self.scene_identity
        we = self.world_environment
        ps = self.procedural_structure
        ts = self.texture_stack
        ae = self.action_event
        tl = self.temporal_layer
        at = self.atmosphere
        li = self.lighting
        cc = self.camera_composition
        sm = self.style_render_mode
        qd = self.quality_detail

        forms_str = ", ".join(ps.procedural_forms)
        neg_str = ", ".join(self.negative_constraints)

        return (
            f"Create a visually stunning procedural scene for the Krystal-Stack universe.\n\n"
            f"Scene identity:\n"
            f"{si.scene_type}, centered on {si.core_theme}, representing {si.narrative_role}.\n\n"
            f"World / environment:\n"
            f"The scene takes place in {we.environment}, at a {we.scale} scale, with {we.terrain_logic}.\n\n"
            f"Procedural structure:\n"
            f"Use {forms_str} (D{ps.dihedral_folds} dihedral folds, fractal dimension {ps.fractal_dimension:.2f}).\n\n"
            f"Texture stack:\n"
            f"Base materials are {ts.base_material}.\n"
            f"Secondary materials include {ts.secondary_material}.\n"
            f"Micro-surface detail should show {ts.micro_surface_detail}.\n"
            f"Edges should behave as {ts.edge_behavior}.\n"
            f"Reflectivity / roughness: {ts.reflectivity_roughness}.\n"
            f"Emission: {ts.emission}.\n"
            f"Atmospheric particles: {ts.atmospheric_particles}.\n\n"
            f"Action / event:\n"
            f"The main event is {ae.primary_event}, caused by {ae.cause}.\n"
            f"Visible transformation: {ae.visible_transformation}.\n"
            f"Environmental response: {ae.environmental_response}.\n"
            f"Motion cues: {ae.motion_cues}.\n"
            f"Narrative traces: {ae.narrative_traces}.\n\n"
            f"Temporal Scene Layer:\n"
            f"Before: {tl.before}\n"
            f"During: {tl.during}\n"
            f"After: {tl.after}\n\n"
            f"Atmosphere:\n"
            f"{at.atmosphere}, with a mood of {at.mood}.\n"
            f"Weather / medium: {at.weather_medium}.\n"
            f"Depth effects: {at.depth_effects}.\n\n"
            f"Lighting:\n"
            f"Primary light: {li.primary_light}.\n"
            f"Secondary light: {li.secondary_light}.\n"
            f"Accent light: {li.accent_light}.\n"
            f"Shadow behavior: {li.shadow_behavior}.\n\n"
            f"Camera / composition:\n"
            f"{cc.camera_angle}, {cc.framing}.\n"
            f"Primary focal point: {cc.primary_focal_point}.\n"
            f"Secondary focal points: {cc.secondary_focal_points}.\n"
            f"Scale references: {cc.scale_references}.\n"
            f"Composition: {cc.composition}.\n\n"
            f"Style / render mode:\n"
            f"{sm.render_mode}, {sm.surface_style}.\n\n"
            f"Quality:\n"
            f"{qd.detail_level}, {qd.texture_richness}, {qd.world_coherence}, {qd.visual_density}.\n\n"
            f"Avoid:\n"
            f"{neg_str}."
        )

    # ── SYNTHESIS: NARRATIVE (DEJOVÝ) PROMPT ────────────────────────────────
    def to_narrative_prompt(self) -> str:
        """
        Synthesizes a continuous, literary, narrative-driven scene description
        focusing heavily on temporal action, world reaction, and layered textures.
        """
        si = self.scene_identity
        we = self.world_environment
        ts = self.texture_stack
        ae = self.action_event
        tl = self.temporal_layer
        li = self.lighting
        cc = self.camera_composition

        return (
            f"V priestore {we.environment} sa odohráva {ae.primary_event}. "
            f"{tl.before} V tomto okamihu {ae.visible_transformation}, pričom {ae.environmental_response}. "
            f"Povrch z {ts.base_material} je popretkávaný {ts.secondary_material}, zatiaľ čo {ts.micro_surface_detail}. "
            f"Z hrán ({ts.edge_behavior}) vyžaruje {ts.emission}, obklopené {ts.atmospheric_particles}. "
            f"Svetelná scéna je komponovaná z {li.primary_light} s akcentmi {li.accent_light}. "
            f"Pohľad z {cc.camera_angle} sústredí pozornosť na {cc.primary_focal_point}, pričom v pozadí {tl.after}."
        )

    # ── SYNTHESIS: ULTRA-COMPACT GENERATOR PROMPT ──────────────────────────
    def to_compact_prompt(self) -> str:
        """
        Synthesizes an ultra-compact, high-density prompt ideal for
        direct token-constrained LLM generation, diffusion models, or agent loops.
        """
        si = self.scene_identity
        we = self.world_environment
        ps = self.procedural_structure
        ts = self.texture_stack
        ae = self.action_event
        li = self.lighting
        sm = self.style_render_mode

        return (
            f"{si.scene_type} ({si.core_theme}) | {we.environment}, {we.scale} scale | "
            f"Forms: {', '.join(ps.procedural_forms[:3])} (D{ps.dihedral_folds}) | "
            f"Textures: {ts.base_material}, {ts.secondary_material}, {ts.micro_surface_detail} | "
            f"Emission: {ts.emission} | Event: {ae.primary_event} ({ae.motion_cues}) | "
            f"Lighting: {li.primary_light}, {li.accent_light} | Style: {sm.render_mode}"
        )

    # ── SYNTHESIS: VULKAN / GODOT 4.x GLSL UNIFORMS ────────────────────────
    def to_godot_shader_uniforms(self) -> Dict[str, Any]:
        """
        Translates procedural structure, folds, and emission parameters
        to Vulkan Forward+ shader constants for Godot 4.x.
        """
        ps = self.procedural_structure
        theme = self.scene_identity.core_theme.lower()

        # Color tensor mappings based on theme
        if "crystal" in theme or "frost" in theme:
            primary_col = [0.0, 0.95, 1.0, 1.0]  # Cyan
            accent_col = [0.83, 0.69, 0.22, 1.0]  # Gold
        elif "toxic" in theme:
            primary_col = [0.22, 1.0, 0.08, 1.0]  # Acid emerald
            accent_col = [0.65, 0.17, 0.95, 1.0]  # Purple miasma
        elif "druid" in theme:
            primary_col = [0.33, 0.42, 0.18, 1.0]  # Forest moss
            accent_col = [0.0, 1.0, 0.53, 1.0]    # Bio-spore green
        elif "alchem" in theme:
            primary_col = [0.83, 0.69, 0.22, 1.0]  # Royal gold
            accent_col = [0.0, 0.95, 1.0, 1.0]    # Alchemical cyan
        elif "holo" in theme or "cyber" in theme:
            primary_col = [0.0, 0.95, 1.0, 1.0]  # Neon cyan
            accent_col = [1.0, 0.0, 0.55, 1.0]    # Magenta
        else:
            primary_col = [1.0, 0.35, 0.1, 1.0]  # Volcanic ember
            accent_col = [0.2, 0.8, 1.0, 1.0]

        return {
            "u_mirror_folds": ps.dihedral_folds,
            "u_fold_angle": round(3.14159265 / max(2, ps.dihedral_folds), 5),
            "u_recursion_limit": min(32, ps.octaves * 4),
            "u_fractal_dimension": ps.fractal_dimension,
            "u_fresnel_factor": 0.88,
            "u_chromatic_aberration": 0.025,
            "u_primary_color": primary_col,
            "u_accent_color": accent_col,
            "vital_max_hp_rule": VITAL_MAX_HP
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_yaml(self) -> str:
        """Lightweight zero-dependency YAML serialization."""
        d = self.to_dict()
        lines = [f"# Krystal-Stack Procedural Visual Template: {self.title}", f"schema_version: '{self.schema_version}'"]
        for k, v in d.items():
            if k == "schema_version":
                continue
            if isinstance(v, dict):
                lines.append(f"{k}:")
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, list):
                        lines.append(f"  {sub_k}:")
                        for item in sub_v:
                            lines.append(f"    - '{item}'")
                    else:
                        lines.append(f"  {sub_k}: '{sub_v}'")
            elif isinstance(v, list):
                lines.append(f"{k}:")
                for item in v:
                    lines.append(f"  - '{item}'")
            else:
                lines.append(f"{k}: '{v}'")
        return "\n".join(lines)


# ==============================================================================
# CANONICAL PRESET REPOSITORY: 10 HIGH-FIDELITY TEMPLATES
# ==============================================================================

def _create_canonical_presets() -> Dict[str, ProceduralVisualTemplate]:
    presets: Dict[str, ProceduralVisualTemplate] = {}

    # 1. CRYSTAL AWAKENING RITUAL
    presets["CRYSTAL_AWAKENING_RITUAL"] = ProceduralVisualTemplate(
        template_id="CRYSTAL_AWAKENING_RITUAL",
        title="Posvätná Procedurálna Aréna: Prebudenie Aéterového Jadra",
        scene_identity=SceneIdentity(
            scene_type="sacred mirror chamber",
            core_theme="crystal",
            narrative_role="awakening"
        ),
        world_environment=WorldEnvironment(
            environment="floating ritual platform suspended above a dark abyss",
            scale="monumental",
            terrain_logic="fractured crystalline plateaus, suspended stone fragments, glowing runic pathways"
        ),
        procedural_structure=ProceduralStructure(
            procedural_forms=[
                "fractal crystal spires",
                "recursive dihedral mirror folds",
                "Voronoi fracture patterns",
                "floating polyhedral debris",
                "spiral rune orbits",
                "volumetric fog bands"
            ],
            fractal_dimension=2.68,
            dihedral_folds=6,
            octaves=6
        ),
        texture_stack=TextureStack(
            base_material="translucent blue crystal and polished obsidian",
            secondary_material="gold alchemical engravings and luminous rune veins",
            micro_surface_detail="fine scratches, tiny ice fractures, shimmering dust particles",
            edge_behavior="chipped edges with glowing cyan highlights",
            reflectivity_roughness="semi-gloss crystal, matte stone, sharp fresnel reflections on polished faces",
            emission="internal cyan core glow and thin golden energy seams",
            atmospheric_particles="drifting frost dust and slow holographic sparks"
        ),
        action_event=ActionEvent(
            primary_event="an ancient crystal core is awakening",
            cause="a sacred alchemical rune sequence has been completed",
            visible_transformation="the central crystal rotates, fractures open across the platform, expanding internal light pulses",
            environmental_response="fog is pushed outward in concentric waves, floating shards rise into the air, golden runes light up in sequence",
            motion_cues="rotating particles, pulsing beams, drifting debris, light distortion in the air",
            narrative_traces="scorched ritual marks, fractured seals, abandoned offerings, traces of previous failed activations"
        ),
        temporal_layer=TemporalLayer(
            before="the ritual chamber remained dormant for centuries beneath frozen stone.",
            during="the crystal core is now awakening, cracking the platform and releasing light through the runic seams.",
            after="the expanding energy field suggests the portal will fully open and reshape the surrounding biome."
        ),
        atmosphere=AtmosphereBlock(
            atmosphere="mystical, tense, sacred and computational",
            mood="awe, latent danger, revelation",
            weather_medium="cold vapor haze, suspended glittering particles",
            depth_effects="layered volumetric fog, distant silhouette fading, atmospheric scattering"
        ),
        lighting=LightingBlock(
            primary_light="soft cyan glow from the crystal core",
            secondary_light="warm golden rune reflections across the platform",
            accent_light="magenta spectral highlights in the mirror recursion",
            shadow_behavior="deep cinematic shadows, sharp silhouette contrast, subtle volumetric shafts"
        ),
        camera_composition=CameraComposition(
            camera_angle="slightly low cinematic perspective",
            framing="wide shot with strong central focus",
            primary_focal_point="the awakening crystal core",
            secondary_focal_points="floating debris, ritual rings, distant portal glow",
            scale_references="tiny shrine pillars and fractured statues",
            composition="asymmetrical editorial composition with large negative space and dramatic center lighting"
        ),
        style_render_mode=StyleRenderMode(
            render_mode="procedural cinematic concept art for Krystal-Stack",
            surface_style="ultra-detailed textured fantasy-tech hybrid"
        )
    )

    # 2. TOXIC BLOOM RUPTURE
    presets["TOXIC_BLOOM_RUPTURE"] = ProceduralVisualTemplate(
        template_id="TOXIC_BLOOM_RUPTURE",
        title="Hnijúca Slatina: Erupcia Slizového Gejzíru",
        scene_identity=SceneIdentity(
            scene_type="biome world",
            core_theme="toxic",
            narrative_role="emergence"
        ),
        world_environment=WorldEnvironment(
            environment="corroded basalt marshland and bubbling acid vents",
            scale="monumental",
            terrain_logic="honeycomb rock dissolution, subterranean acid channels, floating fungal crusts"
        ),
        procedural_structure=ProceduralStructure(
            procedural_forms=[
                "Voronoi slime cavities",
                "spiral spore geysers",
                "branching miasma tendrils",
                "cellular foam membranes"
            ],
            fractal_dimension=2.32,
            dihedral_folds=4,
            octaves=5
        ),
        texture_stack=TextureStack(
            base_material="wet pitted basalt, corroded verdigris bronze, and glistening acid jelly",
            secondary_material="fluorescent emerald mold, chitinous scales, and sulfurous crust",
            micro_surface_detail="bursting micro-bubbles, etched corrosion pits, oily iridescent slick",
            edge_behavior="dissolving ragged stone rims with dripping toxic dew",
            reflectivity_roughness="variable wet gloss on slime, porous ultra-matte on dead volcanic rock",
            emission="bioluminescent lime-green fungal veins and sickly violet bubbling spots",
            atmospheric_particles="drifting toxic pollen, boiling sulfuric steam plumes, flickering spore fireflies"
        ),
        action_event=ActionEvent(
            primary_event="a pressurized subterranean toxic vein ruptures through the stone crust",
            cause="the drainage wards of the old citadel failed under tectonic stress",
            visible_transformation="viscous emerald sludge fountains erupt upwards, dissolving surrounding basalt spires",
            environmental_response="acid mist floods the lowlands, nearby fungal totems swell and release blinding spore bursts",
            motion_cues="violent fluid splashes, billowing green vapor spirals, quivering gelatinous sacs",
            narrative_traces="half-melted iron warding stakes, corroded skeletons of armored sentries, etched acidic runoff gullies"
        ),
        temporal_layer=TemporalLayer(
            before="pressure built silently beneath calcified sediment and abandoned drainage canals.",
            during="a massive emerald eruption blasts through the floor, shattering bedrock into dissolving fragments.",
            after="the acid marsh will submerge the low ground, transforming the sector into an uninhabitable miasma swamp."
        ),
        atmosphere=AtmosphereBlock(
            atmosphere="oppressive, noxious, primeval, and predatory",
            mood="dread, claustrophobia, toxic beauty",
            weather_medium="thick yellowish sulfur smog and steady acid drizzle",
            depth_effects="heavy green depth fog obscuring horizon, silhouetted dead monoliths"
        ),
        lighting=LightingBlock(
            primary_light="intense lime-green glow emanating from the erupting fissure",
            secondary_light="deep violet bioluminescence from surrounding fungal nodes",
            accent_light=" sickly yellow rim light cutting through the sulfurous smoke",
            shadow_behavior="murky diffused shadows softened by dense atmospheric haze"
        ),
        camera_composition=CameraComposition(
            camera_angle="ground-level upward tilt",
            framing="dramatic medium-wide framing capturing the volcanic plume",
            primary_focal_point="the geyser eruption apex",
            secondary_focal_points="corroded watchtower ruins, expanding acid ring",
            scale_references="decayed wooden barricades and broken iron siege shields",
            composition="dynamic diagonal energy vector with bubbling foreground details"
        ),
        style_render_mode=StyleRenderMode(
            render_mode="procedural cinematic concept art for Krystal-Stack",
            surface_style="gritty bio-industrial dark fantasy"
        )
    )

    # 3. DRUID WORLD TREE RESONANCE
    presets["DRUID_WORLD_TREE_RESONANCE"] = ProceduralVisualTemplate(
        template_id="DRUID_WORLD_TREE_RESONANCE",
        title="Pradávny Les: Rezonancia Prvotného Stromu Sveta",
        scene_identity=SceneIdentity(
            scene_type="holographic shrine",
            core_theme="druid",
            narrative_role="resonance"
        ),
        world_environment=WorldEnvironment(
            environment="primeval monolithic forest clearing centered on colossal gnarled roots",
            scale="cathedral-scale",
            terrain_logic="ancient moss-covered flagstones lifted by spiraling heartwood roots, sunken springs"
        ),
        procedural_structure=ProceduralStructure(
            procedural_forms=[
                "L-system recursive root lattices",
                "golden ratio phyllotaxis leaf spirals",
                "volumetric amber canopy godrays",
                "mycorrhizal energy pulses"
            ],
            fractal_dimension=2.85,
            dihedral_folds=8,
            octaves=7
        ),
        texture_stack=TextureStack(
            base_material="ancient ironwood bark, dark river silt, and petrified moss",
            secondary_material="golden sap inlays, glowing mycelial filaments, and carved oak runes",
            micro_surface_detail="deep fibrous bark furrows, dew-laden lichen fronds, micro-spore clusters",
            edge_behavior="overgrown organic softness with trailing vine tendrils",
            reflectivity_roughness="velvety matte moss, damp reflective wood grain, glass-like amber droplets",
            emission="warm golden amber sap currents and pulsing cyan mycorrhizal networks",
            atmospheric_particles="floating luminescent spores, descending autumn needles, drifting pollen haze"
        ),
        action_event=ActionEvent(
            primary_event="the heart of the World Tree beats with a harmonic restoration pulse",
            cause="the seasonal alignment of celestial constellations has converged at zenith",
            visible_transformation="ancient bark splits open to reveal an undulating amber heart; dormant roots surge with emerald light",
            environmental_response="withered flora instantly rejuvenates, stone monoliths right themselves, spring water rises in crystal pools",
            motion_cues="pulsing root veins, floating spores orbiting in harmonic circles, leaves shivering with energy",
            narrative_traces="abandoned druidic carved bells, worn stone prayer circles, root-entangled tribal crests"
        ),
        temporal_layer=TemporalLayer(
            before="the grove lay silent in petrified slumber, waiting for the equinox alignment.",
            during="a massive pulse of vital amber energy radiates from the trunk, reversing centuries of decay.",
            after="the awakened forest network will spread new growth across adjacent sectors, binding the earth."
        ),
        atmosphere=AtmosphereBlock(
            atmosphere="serene, ancient, awe-inspiring, and vital",
            mood="reverence, renewal, spiritual power",
            weather_medium="warm vapor mist sparkling with bioluminescent motes",
            depth_effects="dense tree trunk silhouettes receding into golden atmospheric glow"
        ),
        lighting=LightingBlock(
            primary_light="warm golden amber glow from the central tree heart",
            secondary_light="diffuse cyan radiance from the ground root network",
            accent_light="sunlit godrays filtering through the towering canopy",
            shadow_behavior="soft organic dappled shadows with high micro-contrast"
        ),
        camera_composition=CameraComposition(
            camera_angle="sweeping low-angle upward perspective",
            framing="majestic vertical composition capturing the tree trunk ascension",
            primary_focal_point="the glowing amber heartwood core",
            secondary_focal_points="radiating surface roots, ancient standing stones",
            scale_references="tiny carved offerings, human-scale shrine steps",
            composition="symmetrical cathedral framing balanced by natural organic asymmetry"
        ),
        style_render_mode=StyleRenderMode(
            render_mode="procedural cinematic concept art for Krystal-Stack",
            surface_style="richly textured mythic realism"
        )
    )

    # 4. HOLOGRAPHIC ASCII MIRROR CHAMBER
    presets["HOLOGRAPHIC_ASCII_MIRROR_CHAMBER"] = ProceduralVisualTemplate(
        template_id="HOLOGRAPHIC_ASCII_MIRROR_CHAMBER",
        title="Rekurzívna AR Sála: Kvantový ASCII Manifold D16",
        scene_identity=SceneIdentity(
            scene_type="ASCII hologram landscape",
            core_theme="cyber-ritual",
            narrative_role="transformation"
        ),
        world_environment=WorldEnvironment(
            environment="infinite mirrored computational void crossed by optical beam waveguides",
            scale="infinite",
            terrain_logic="quantized grid planes, floating monospace matrices, dihedrally folded mirror planes"
        ),
        procedural_structure=ProceduralStructure(
            procedural_forms=[
                "D16 dihedral kaleidoscopic IFS folds",
                "stereoscopic anaglyph beam splittings",
                "character cell glyph lattices",
                "Schlick-Fresnel ray rebounds"
            ],
            fractal_dimension=3.12,
            dihedral_folds=16,
            octaves=8
        ),
        texture_stack=TextureStack(
            base_material="black optical glass, liquid mercury floors, and carbon nanotube grids",
            secondary_material="luminous glyph tracks, scanline interference fringes, and wireframe edges",
            micro_surface_detail="nanoscale etched circuit traces, laser raster lines, CRT phosphors",
            edge_behavior="knife-sharp pixelated vector borders glowing with laser intensity",
            reflectivity_roughness="perfect mirror reflectance (roughness 0.02) with infinite recursive echoes",
            emission="blinding neon cyan (632nm) and hot magenta stereoscopic fringe lines",
            atmospheric_particles="drifting floating punctuation glyphs, laser dust, quantized photons"
        ),
        action_event=ActionEvent(
            primary_event="the dihedral reflection planes fold inward, compressing real-time geometry into ASCII tensors",
            cause="the Antigravity engine initiated a matrix quantization pass across the 3D scene",
            visible_transformation="3D surfaces dissolve into glowing matrix characters, repeating into kaleidoscopic infinity",
            environmental_response="coordinate grids ripple, light rays split into cyan and red chromatic streams",
            motion_cues="rapidly cascading character falls, oscillating scanlines, folding geometric planes",
            narrative_traces="fragmented memory registers, frozen terminal prompt lines, disconnected telemetry nodes"
        ),
        temporal_layer=TemporalLayer(
            before="the void was a silent flat reflection plane awaiting algorithmic input.",
            during="the D16 dihedral folds expand, multiplying every light ray into 16 kaleidoscopic iterations.",
            after="the entire scene stabilizes into an interactive real-time ASCII holographic manifold."
        ),
        atmosphere=AtmosphereBlock(
            atmosphere="hyper-technological, cyber-spiritual, analytical, and boundless",
            mood="intellectual transcendence, dizzying scale, digital mysticism",
            weather_medium="dark vacuum crossed by optical beam haze and phosphor glow",
            depth_effects="infinite recursive depth fading into black singularity, stereoscopic parallax"
        ),
        lighting=LightingBlock(
            primary_light="coherent cyan laser collimators at 632.8 nm",
            secondary_light="magenta chromatic dispersion fringes bouncing off mirror edges",
            accent_light="sharp amber cursor pings at matrix nodal intersections",
            shadow_behavior="absence of shadows; illumination determined strictly by ray bounce depth"
        ),
        camera_composition=CameraComposition(
            camera_angle="axial alignment along the primary kaleidoscopic symmetry line",
            framing="centered infinite tunnel perspective",
            primary_focal_point="the central convergence singularity",
            secondary_focal_points="multiplying peripheral mirror quadrants",
            scale_references="floating monospace characters forming scale dimensions",
            composition="perfect radial kaleidoscopic symmetry with intense peripheral repetition"
        ),
        style_render_mode=StyleRenderMode(
            render_mode="holographic ASCII Vulkan Forward+ simulation",
            surface_style="laser-etched retro-futuristic wireframe"
        )
    )

    # 5. MONASTIC ALCHEMICAL TRANSMUTATION
    presets["MONASTIC_ALCHEMICAL_TRANSMUTATION"] = ProceduralVisualTemplate(
        template_id="MONASTIC_ALCHEMICAL_TRANSMUTATION",
        title="Alchymistické Laboratórium: Zlatý Rez & Dihedrálny Kruh",
        scene_identity=SceneIdentity(
            scene_type="ancient ritual",
            core_theme="alchemy",
            narrative_role="transformation"
        ),
        world_environment=WorldEnvironment(
            environment="subterranean vaulted cloister with concentric brass alchemical rings",
            scale="cathedral-scale",
            terrain_logic="concentric stone rings inlayed with golden molten canals, rotating planetary dials"
        ),
        procedural_structure=ProceduralStructure(
            procedural_forms=[
                "D12 dodecahedral symmetry plates",
                "nested Archimedean spirals",
                "golden ratio interlocking gears",
                "flowing liquid gold troughs"
            ],
            fractal_dimension=2.5,
            dihedral_folds=12,
            octaves=6
        ),
        texture_stack=TextureStack(
            base_material="hewn basalt blocks, oxidized dark bronze, and carved Carrara marble",
            secondary_material="heavy 24k gold inlays, quicksilver channels, and cinnabar powder",
            micro_surface_detail="chiseled latin inscriptions, chisel marks, oil stains, powdered sulfur dust",
            edge_behavior="smoothly burnished gold-filled bevels, worn stone steps",
            reflectivity_roughness="deep luster on polished brass, matte absorbent sandstone, liquid mercury sheen",
            emission="molten golden incandescent glow from floor channels, faint emerald crucible flames",
            atmospheric_particles="curling incense smoke, glowing amber embers, vaporized mercury fumes"
        ),
        action_event=ActionEvent(
            primary_event="the planetary gears lock into alignment, initiating the Great Alchemical Work",
            cause="the philosopher's tincture was poured into the central mercury crucible",
            visible_transformation="base lead slabs begin recrystallizing into brilliant faceted gold along runic seams",
            environmental_response="rotating brass rings accelerate, humming in harmonic fifths, heat waves warp the air",
            motion_cues="smooth mechanical ring rotation, bubbling molten metals, rising smoke spirals",
            narrative_traces="wax-sealed grimoires, discarded glass alembics, failed blackened transmutation ingots"
        ),
        temporal_layer=TemporalLayer(
            before="monks chanted for three days while tending the coal fires beneath the dials.",
            during="the alignment clicks into place, and golden veins rapidly spread through solid rock.",
            after="the entire chamber will stabilize as an indestructible golden celestial observatory."
        ),
        atmosphere=AtmosphereBlock(
            atmosphere="reverent, scholastic, heavy, and incandescent",
            mood="illumination, sacred order, profound secrecy",
            weather_medium="warm indoor atmosphere thick with frankincense and ozone",
            depth_effects="vaulted archways receding into amber candlelight, deep atmospheric shadows"
        ),
        lighting=LightingBlock(
            primary_light="radiant golden luminescence from the central transmutation pit",
            secondary_light="flickering tallow candles lining the tiered stone gallery",
            accent_light="green reflection off alchemical glass alembics",
            shadow_behavior="dramatic chiaroscuro with sharp architectural shadow silhouettes"
        ),
        camera_composition=CameraComposition(
            camera_angle="elevated balcony three-quarter isometric overview",
            framing="comprehensive geometric composition showing all concentric rings",
            primary_focal_point="the central golden crucible and aligning dials",
            secondary_focal_points="scholastic scriptorium niches, steam exhaust vents",
            scale_references="human-sized lecterns and tiered amphitheater seating",
            composition="harmonious Fibonacci spiral composition following the golden ratio"
        ),
        style_render_mode=StyleRenderMode(
            render_mode="procedural cinematic concept art for Krystal-Stack",
            surface_style="Renaissance dark fantasy with hyper-precise mechanical detailing"
        )
    )

    # 6. SUBTERRANEAN OBSIDIAN FORGE
    presets["SUBTERRANEAN_OBSIDIAN_FORGE"] = ProceduralVisualTemplate(
        template_id="SUBTERRANEAN_OBSIDIAN_FORGE",
        title="Vulkanická Huta: Kováčňa Magmatického Jadra",
        scene_identity=SceneIdentity(
            scene_type="procedural arena",
            core_theme="volcanic",
            narrative_role="collapse"
        ),
        world_environment=WorldEnvironment(
            environment="cavernous volcanic caldera bridged by black basalt arches over molten lava",
            scale="monumental",
            terrain_logic="solidified obsidian crust over churning magma, columnar basalt pillars"
        ),
        procedural_structure=ProceduralStructure(
            procedural_forms=[
                "columnar basalt hex prisms",
                "fluid dynamic lava convection cells",
                "cooling crack networks",
                "volcanic ash billows"
            ],
            fractal_dimension=2.4,
            dihedral_folds=6,
            octaves=5
        ),
        texture_stack=TextureStack(
            base_material="glassy black obsidian, porous volcanic scoria, and incandescent slag",
            secondary_material="white-hot basalt crust, sulfur seams, and forged blackened iron rivets",
            micro_surface_detail="cooling fissures glowing red, conchoidal fractures on obsidian, gritty ash layers",
            edge_behavior="razor-sharp fractured glass rims with orange thermal glow",
            reflectivity_roughness="mirror-smooth glassy obsidian faces contrasted with rough matte slag",
            emission="intense magma glow (1200°C) ranging from brilliant yellow-white to deep crimson",
            atmospheric_particles="rising sparks, swirling soot devils, sulfur fumes"
        ),
        action_event=ActionEvent(
            primary_event="a tectonic surge destabilizes the cooling magma floor",
            cause="the geothermal pressure taps were overloaded by the automated defense forge",
            visible_transformation="columnar basalt bridges buckle and plunge into molten rock; geysers of liquid fire erupt",
            environmental_response="dense ash clouds blacken the cavern ceiling, heat tremors crack the forge anvils",
            motion_cues="surging lava waves, falling columnar pillars, violent steam bursts",
            narrative_traces="half-forged titanium war-hammers abandoned on anvils, scorched protective golems"
        ),
        temporal_layer=TemporalLayer(
            before="the forge operated at peak efficiency, casting siege weapons for the citadel.",
            during="a sudden thermal surge ruptures the retaining dam, flooding the lower staging arenas.",
            after="the entire lower complex will cool into a jagged maze of sharp volcanic glass."
        ),
        atmosphere=AtmosphereBlock(
            atmosphere="oppressive, scorching, cataclysmic, and raw",
            mood="immediate peril, brutal power, volcanic majesty",
            weather_medium="suffocating heat haze and heavy falling black cinders",
            depth_effects="silhouette pillars fading through glowing orange thermal haze"
        ),
        lighting=LightingBlock(
            primary_light="blinding under-lighting from the lava sea",
            secondary_light="cool blue skylight from a fractured ceiling chimney far above",
            accent_light="crimson rim reflections on glossy obsidian columns",
            shadow_behavior="stark upward-cast shadows accentuating jagged architectural edges"
        ),
        camera_composition=CameraComposition(
            camera_angle="dramatic low-angle shot from the edge of the collapsing walkway",
            framing="wide expansive view showcasing the colossal vertical drop",
            primary_focal_point="the collapsing central forge bridge",
            secondary_focal_points="bubbling lava whirlpool, intact hanging observation cages",
            scale_references="giant industrial chains and human-scaled maintenance ladders",
            composition="strong vertical and diagonal lines emphasizing collapse and kinetic power"
        ),
        style_render_mode=StyleRenderMode(
            render_mode="procedural cinematic concept art for Krystal-Stack",
            surface_style="dark atmospheric industrial fantasy"
        )
    )

    # 7. AERIAL FLOATING SPIRE CITADEL
    presets["AERIAL_FLOATING_SPIRE_CITADEL"] = ProceduralVisualTemplate(
        template_id="AERIAL_FLOATING_SPIRE_CITADEL",
        title="Nebeská Citadela: Levitujúce Aéterové Veže",
        scene_identity=SceneIdentity(
            scene_type="citadel bastion",
            core_theme="astral",
            narrative_role="stabilization"
        ),
        world_environment=WorldEnvironment(
            environment="sky archipelago of inverted limestone islands floating above a cloud sea",
            scale="cathedral-scale",
            terrain_logic="inverted conical rock masses, arched aether bridges, cascading zero-g waterfalls"
        ),
        procedural_structure=ProceduralStructure(
            procedural_forms=[
                "inverted signed distance field gravity cones",
                "catenary hanging chain networks",
                "streamline aerodynamic spires",
                "cloud turbulence vortexes"
            ],
            fractal_dimension=2.6,
            dihedral_folds=8,
            octaves=6
        ),
        texture_stack=TextureStack(
            base_material="sun-bleached marble, blue slate roof shingles, and magnetic aetherite ore",
            secondary_material="polished brass aeronautical compasses, woven silk banners, and sapphire glass",
            micro_surface_detail="wind-scoured stone channels, delicate filigree railings, cloud moisture beads",
            edge_behavior="crisp architectural moldings, sheer vertical cliff drop-offs",
            reflectivity_roughness="smooth reflective marble pavements, soft diffuse clouds, gleaming metal domes",
            emission="soft azure gravity-repulsion fields under each floating island keel",
            atmospheric_particles="drifting cloud wisps, fluttering fabric ribbons, floating dandelion fluff"
        ),
        action_event=ActionEvent(
            primary_event="the celestial alignment stabilizes the magnetic keels of the sky citadel",
            cause="the central aether reactor was re-ignited after an aerial siege",
            visible_transformation="the floating islands align into a fortified defensive ring; gravitational bridges lock together",
            environmental_response="wind shear calms into gentle updrafts, clouds form a protective barrier below",
            motion_cues="graceful island bobbing, rotating observation rings, rushing water breaking into spray",
            narrative_traces="docked aerial skiffs, patched ballista battlements, fluttering victor pennants"
        ),
        temporal_layer=TemporalLayer(
            before="the archipelago drifted aimlessly during a catastrophic gravity core failure.",
            during="the blue repulsion coils fire up, locking the massive rock islands into a defensive lattice.",
            after="the citadel will serve as an unassailable bastion overlooking the lower biomes."
        ),
        atmosphere=AtmosphereBlock(
            atmosphere="weightless, triumphant, ethereal, and pristine",
            mood="liberation, serene vigilance, boundless horizon",
            weather_medium="crystal-clear thin stratosphere with fast-moving cirrus clouds",
            depth_effects="distant floating islands fading into brilliant atmospheric aerial perspective"
        ),
        lighting=LightingBlock(
            primary_light="brilliant golden hour sunlight streaming across the cloud tops",
            secondary_light="cool blue sky dome fill light",
            accent_light="azure emission from the anti-gravity rings underneath the islands",
            shadow_behavior="sharp clean shadows cast across cloud banks far below"
        ),
        camera_composition=CameraComposition(
            camera_angle="wide aerial panorama from a flying perspective",
            framing="epic panoramic establishing shot with open horizon",
            primary_focal_point="the central grand spire and its glowing keel",
            secondary_focal_points="connecting sky bridges, smaller satellite outposts",
            scale_references="circling winged scouts and tiny wind turbine sails",
            composition="rule-of-thirds composition balancing solid island mass with infinite sky"
        ),
        style_render_mode=StyleRenderMode(
            render_mode="procedural cinematic concept art for Krystal-Stack",
            surface_style="clean luminous high fantasy with hard-surface sci-fi precision"
        )
    )

    # 8. BIOMECHANICAL XENOBIOTIC HIVE
    presets["BIOMECHANICAL_XENOBIOTIC_HIVE"] = ProceduralVisualTemplate(
        template_id="BIOMECHANICAL_XENOBIOTIC_HIVE",
        title="Xenobiotické Hniezdo: Gigerovský Biomechanický Labyrint",
        scene_identity=SceneIdentity(
            scene_type="biome world",
            core_theme="biomechanical",
            narrative_role="invasion"
        ),
        world_environment=WorldEnvironment(
            environment="alien hive interior constructed from ribbed chitin and fused pneumatic bone arches",
            scale="monumental",
            terrain_logic="ribbed tubular corridors, biomechanical valves, translucent incubation sacs"
        ),
        procedural_structure=ProceduralStructure(
            procedural_forms=[
                "hyperbolic gyroid bio-surfaces",
                "recursive spinal vertebrae columns",
                "biomechanical tendon cables",
                "periodic ventilation sphincters"
            ],
            fractal_dimension=2.92,
            dihedral_folds=8,
            octaves=6
        ),
        texture_stack=TextureStack(
            base_material="burnished black chitin, biomechanical bone resin, and cold blued steel",
            secondary_material="translucent amniotic slime, bronze hydraulic pipes, and pulsating neural cords",
            micro_surface_detail="fingerprint-like ridge patterns, glistening mucus films, microscopic respiratory pores",
            edge_behavior="segmented carapace overlaps with razor-thin chitinous flanges",
            reflectivity_roughness="high gloss wet slime coating over satin dark chitin",
            emission="faint ghostly bioluminescent orange pulses running along neural pipelines",
            atmospheric_particles="suspended condensation mist, drifting chitinous husks, pulsating spore puffs"
        ),
        action_event=ActionEvent(
            primary_event="the hive intelligence initiates a synchronized gestation cycle",
            cause="an organic telemetry signal was broadcast from an encroaching surface scout",
            visible_transformation="incubation pods twitch and inflate, wall tendons contract, opening ribbed airlocks",
            environmental_response="a low-frequency sub-bass vibration shakes the floor; amniotic fluid cascades down wall conduits",
            motion_cues="rhythmic breathing peristalsis of the walls, sliding tendon cables, rising vapor puffs",
            narrative_traces="fused cybernetic implants of fallen explorers, calcified cocoons, disconnected data cables"
        ),
        temporal_layer=TemporalLayer(
            before="the hive slumbered in anaerobic stasis inside deep subterranean basalt crevices.",
            during="thousands of bio-mechanical pods hum to life as neural cables flush with glowing nutrient fluid.",
            after="swarms of xenobiotic drones will deploy through ventilation shafts toward the surface sectors."
        ),
        atmosphere=AtmosphereBlock(
            atmosphere="claustrophobic, alien, relentless, and synthetic-organic",
            mood="visceral horror, dark elegance, hypnotic biomechanical order",
            weather_medium="warm humid bio-mist smelling of ozone and sweet amniotic fluid",
            depth_effects="endless repeating spinal ribs fading into dark vanishing points"
        ),
        lighting=LightingBlock(
            primary_light="cold surgical white back-light filtering through ribbed wall slits",
            secondary_light="pulsing internal amber glow from incubation membranes",
            accent_light="slight green specular reflection on wet chitinous crests",
            shadow_behavior="deep pitch-black shadows concealing intricate mechanical crevasses"
        ),
        camera_composition=CameraComposition(
            camera_angle="first-person exploratory corridor angle with slight camera tilt",
            framing="claustrophobic interior framing emphasizing enclosing rib arches",
            primary_focal_point="a massive translucent queen incubation membrane at the corridor terminus",
            secondary_focal_points="hanging hydraulic cables, wall-mounted biomechanical sentries",
            scale_references="humanoid bio-armor suits fused into wall resin",
            composition="intense one-point perspective tunnel composition creating overwhelming depth"
        ),
        style_render_mode=StyleRenderMode(
            render_mode="procedural cinematic concept art for Krystal-Stack",
            surface_style="hyper-detailed Giger-esque biomechanical surrealism"
        )
    )

    # 9. FROST GLACIAL FRACTURE CHASM
    presets["FROST_GLACIAL_FRACTURE_CHASM"] = ProceduralVisualTemplate(
        template_id="FROST_GLACIAL_FRACTURE_CHASM",
        title="Severné Štíty: Ľadovcová Trhlina Kryštálového Kmeňa",
        scene_identity=SceneIdentity(
            scene_type="biome world",
            core_theme="frost",
            narrative_role="transformation"
        ),
        world_environment=WorldEnvironment(
            environment="abyssal glacial canyon flanked by translucent ice cliffs and frozen water needles",
            scale="monumental",
            terrain_logic="crevasse step walls, shear stress fractures, hanging blue seracs, frozen waterfalls"
        ),
        procedural_structure=ProceduralStructure(
            procedural_forms=[
                "ridged mountain manifolds",
                "stress tensor brittle fractures",
                "hexagonal columnar ice needles",
                "volumetric katabatic wind flows"
            ],
            fractal_dimension=2.55,
            dihedral_folds=6,
            octaves=6
        ),
        texture_stack=TextureStack(
            base_material="dense cyan glacial ice, compacted firn snow, and dark frozen shale",
            secondary_material="rhomboid frost crystals, pure silver rune engravings, and frozen air bubbles",
            micro_surface_detail="hairline internal thermal stress cracks, frosted feathery dendritic patterns",
            edge_behavior="jagged razor ice edges with internal total reflection glow",
            reflectivity_roughness="subsurface scattering translucent ice (roughness 0.08), powdery snow (roughness 0.8)",
            emission="deep cyan subsurface volumetric scattering and pale violet frost runes",
            atmospheric_particles="rushing snow spindrift, glittering diamond dust, howling gale vapor"
        ),
        action_event=ActionEvent(
            primary_event="a massive glacial rift cleaves the valley under the weight of an aether strike",
            cause="the Crystal Tribe channeled their orbital hyper-lance to repel an invading column",
            visible_transformation="a sheer 200-meter ice cliff shears away, collapsing into the chasm with thunderous resonance",
            environmental_response="shockwaves fling blinding snow sheets into the air; exposed deep ice begins glowing with latent mana",
            motion_cues="tumbling ice blocks, billowing snow powder avalanches, shimmering refraction waves",
            narrative_traces="half-frozen siege catapults trapped in ice, ancient tribal banners encased in clear frost"
        ),
        temporal_layer=TemporalLayer(
            before="the glacier stood motionless for millennia, acting as an impassable defensive rampart.",
            during="the orbital beam shatters the ice shelf, opening a yawning chasm that exposes ancient subterranean caverns.",
            after="the rift will become a fortified icy canyon defended by crystal frost wards."
        ),
        atmosphere=AtmosphereBlock(
            atmosphere="bone-chilling, vast, crystalline, and lethal",
            mood="monumental solitude, primal cold, razor clarity",
            weather_medium="blinding blizzard squall transitioning into clear frozen air",
            depth_effects="deep atmospheric indigo Rayleigh scattering across endless snowy peaks"
        ),
        lighting=LightingBlock(
            primary_light="brilliant low-angle arctic sun reflecting blindingly off snow crests",
            secondary_light="deep blue sky dome illumination filling the deep chasm shadows",
            accent_light="turquoise subsurface glow radiating from within the thick ice walls",
            shadow_behavior="crisp cobalt-blue shadows stretching across the fractured snow fields"
        ),
        camera_composition=CameraComposition(
            camera_angle="dramatic edge-of-the-abyss high perspective looking down into the fissure",
            framing="vertically dominant composition emphasizing sheer verticality",
            primary_focal_point="the glowing exposed aether core at the bottom of the chasm",
            secondary_focal_points="collapsing serac towers, distant razor mountain ridges",
            scale_references="tiny fortified ice observation towers on the cliff crest",
            composition="stark diagonal cleavage line dividing the frame into light and shadow halves"
        ),
        style_render_mode=StyleRenderMode(
            render_mode="procedural cinematic concept art for Krystal-Stack",
            surface_style="hyper-realistic cold climate physical rendering"
        )
    )

    # 10. CALABI YAU QUANTUM PORTAL
    presets["CALABI_YAU_QUANTUM_PORTAL"] = ProceduralVisualTemplate(
        template_id="CALABI_YAU_QUANTUM_PORTAL",
        title="Kvantová Singularita: Calabi-Yau 6D Dimenzionálny Portál",
        scene_identity=SceneIdentity(
            scene_type="sacred mirror chamber",
            core_theme="quantum",
            narrative_role="portal activation"
        ),
        world_environment=WorldEnvironment(
            environment="spherical zero-gravity containment dome floating in hyper-dimensional space",
            scale="monumental",
            terrain_logic="curving magnetic containment rings, floating geometric anchors, spacetime ripples"
        ),
        procedural_structure=ProceduralStructure(
            procedural_forms=[
                "Calabi-Yau 6D cross-section manifolds",
                "gravitational lensing event horizons",
                "hyperbolic Riemann sheets",
                "Planck-scale particle torrents"
            ],
            fractal_dimension=3.4,
            dihedral_folds=16,
            octaves=8
        ),
        texture_stack=TextureStack(
            base_material="superconducting dark titanium, polarized diamond windows, and magnetic liquid flux",
            secondary_material="gold quantum stabilization filaments, laser interferometry grids",
            micro_surface_detail="spacetime distortion ripples, micro-gravitational lensing rings, atomic lattice burns",
            edge_behavior="visually bent curved edges warped by intense gravitational refraction",
            reflectivity_roughness="specular polarized mirrors with complex anti-reflective multi-layer coatings",
            emission="brilliant violet-white singularity glow and rainbow synchrotron radiation rings",
            atmospheric_particles="high-energy Cherenkov sparks, tachyon trails, subatomic particle bursts"
        ),
        action_event=ActionEvent(
            primary_event="the 6-dimensional Calabi-Yau manifold unfolds into observable three-dimensional space",
            cause="the singularity containment harmonics achieved absolute zero resonance",
            visible_transformation="the center of the portal twists into an impossible multi-dimensional geometric knot; space curves around it",
            environmental_response="magnetic containment rings glow red-hot; sparks arc across superconducting pylons as gravity reverses",
            motion_cues="violent twisting dimensional folds, orbiting particle halos, light bending into rings",
            narrative_traces="overloaded sensor arrays, melted calibration rods, warning runes inscribed into shielding"
        ),
        temporal_layer=TemporalLayer(
            before="containment magnetic fields strained against the micro-singularity's pull for months.",
            during="the manifold fully blooms into 3D spacetime, bending light and tearing the fabric of reality.",
            after="the dimensional wormhole will link the Krystal-Stack universe to distant parallel realms."
        ),
        atmosphere=AtmosphereBlock(
            atmosphere="mind-bending, mathematically sublime, dangerous, and epochal",
            mood="existential awe, technological triumph, dimensional vertigo",
            weather_medium="vacuum distortion wave with intense gravitational lens shimmer",
            depth_effects="infinite warped background reflections, Doppler-shifted distant stars"
        ),
        lighting=LightingBlock(
            primary_light="blinding ultraviolet-white radiance from the manifold core",
            secondary_light="sapphire Cherenkov radiation washing over the containment hull",
            accent_light="orange warning strobes pulsing rhythmically along the containment perimeter",
            shadow_behavior="gravitationally bent shadows wrapping around curved space"
        ),
        camera_composition=CameraComposition(
            camera_angle="centered axial view looking directly into the singularity aperture",
            framing="monumental symmetrical circular composition",
            primary_focal_point="the undulating 6D Calabi-Yau manifold core",
            secondary_focal_points="glowing magnetic containment pylons, warping background grid",
            scale_references="tiny observation platforms with safety railings on the outer ring",
            composition="concentric circular composition drawing the eye relentlessly into the center"
        ),
        style_render_mode=StyleRenderMode(
            render_mode="procedural cinematic concept art for Krystal-Stack",
            surface_style="hard sci-fi quantum physics visualization blended with cosmic myth"
        )
    )

    return presets


class KrystalProceduralVisualTemplateSystem:
    """
    Core engine managing the Krystal Procedural Visual Template System v1.0.
    Provides template retrieval, prompt compilation (Master, Narrative, Compact),
    Godot uniform translation, and schema validation.
    """

    def __init__(self, templates_dir: Optional[str] = None):
        self.templates_dir = templates_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "templates", "visual_procedural"
        )
        os.makedirs(self.templates_dir, exist_ok=True)
        self._presets: Dict[str, ProceduralVisualTemplate] = _create_canonical_presets()
        self._sync_presets_to_disk()

    def _sync_presets_to_disk(self):
        """Writes canonical presets to disk in JSON & YAML format for artists & tools."""
        for tpl_id, tpl in self._presets.items():
            json_path = os.path.join(self.templates_dir, f"{tpl_id.lower()}.json")
            yaml_path = os.path.join(self.templates_dir, f"{tpl_id.lower()}.yaml")
            try:
                if not os.path.exists(json_path):
                    with open(json_path, "w", encoding="utf-8") as f:
                        f.write(tpl.to_json(indent=2))
                if not os.path.exists(yaml_path):
                    with open(yaml_path, "w", encoding="utf-8") as f:
                        f.write(tpl.to_yaml())
            except Exception:
                pass

    def get_template(self, template_id: str) -> Optional[ProceduralVisualTemplate]:
        return self._presets.get(template_id.upper())

    def list_templates(self) -> List[Dict[str, Any]]:
        return [
            {
                "template_id": t.template_id,
                "title": t.title,
                "scene_type": t.scene_identity.scene_type,
                "core_theme": t.scene_identity.core_theme,
                "narrative_role": t.scene_identity.narrative_role,
                "environment": t.world_environment.environment,
                "dihedral_folds": t.procedural_structure.dihedral_folds,
                "vital_max_hp_rule": t.vital_max_hp_rule
            }
            for t in self._presets.values()
        ]

    def compile_prompt(self, template_id: str, mode: str = "master") -> Dict[str, Any]:
        """
        Compiles a template into the specified output prompt mode:
        - 'master': Official Krystal Master Prompt
        - 'narrative': Literary, dej-rich descriptive prompt
        - 'compact': High-density token-optimized prompt for agents
        - 'all': All formats + Godot Vulkan shader uniforms
        """
        tpl = self.get_template(template_id)
        if not tpl:
            raise KeyError(f"Template '{template_id}' not found in canonical registry.")

        master_p = tpl.to_master_prompt()
        narrative_p = tpl.to_narrative_prompt()
        compact_p = tpl.to_compact_prompt()
        glsl_u = tpl.to_godot_shader_uniforms()

        if mode == "master":
            res = {"prompt": master_p}
        elif mode == "narrative":
            res = {"prompt": narrative_p}
        elif mode == "compact":
            res = {"prompt": compact_p}
        else:
            res = {
                "master_prompt": master_p,
                "narrative_prompt": narrative_p,
                "compact_prompt": compact_p,
                "godot_shader_uniforms": glsl_u
            }

        res["template_id"] = tpl.template_id
        res["title"] = tpl.title
        res["vital_max_hp_rule"] = VITAL_MAX_HP
        return res

    def match_template_by_intent(self, user_text: str) -> ProceduralVisualTemplate:
        """
        Matches a natural language prompt against the 10 canonical presets
        based on theme, environment, and procedural structure keywords.
        """
        t_low = user_text.lower()

        scores: Dict[str, int] = {tid: 0 for tid in self._presets}

        # Keyword mapping rules
        rules = {
            "CRYSTAL_AWAKENING_RITUAL": ["crystal", "kryštál", "aéter", "aether", "cyan", "mirror arena", "prebudenie", "awakening"],
            "TOXIC_BLOOM_RUPTURE": ["toxic", "toxick", "sliz", "acid", "kyslý", "geyser", "spore", "erupcia", "marsh"],
            "DRUID_WORLD_TREE_RESONANCE": ["druid", "strom sveta", "world tree", "les", "forest", "roots", "korene", "moss", "mach", "sap"],
            "HOLOGRAPHIC_ASCII_MIRROR_CHAMBER": ["ascii", "hologram", "holographic", "matrix", "scanline", "d16", "vulkan", "wireframe"],
            "MONASTIC_ALCHEMICAL_TRANSMUTATION": ["alchemy", "alchýmia", "alchemical", "gold", "zlato", "d12", "monastic", "quicksilver"],
            "SUBTERRANEAN_OBSIDIAN_FORGE": ["obsidian", "lava", "volcanic", "vulkán", "forge", "huta", "magma", "slag"],
            "AERIAL_FLOATING_SPIRE_CITADEL": ["aerial", "floating", "island", "ostrov", "citadel", "sky", "nebeská", "aetherite"],
            "BIOMECHANICAL_XENOBIOTIC_HIVE": ["biomechanical", "biomech", "giger", "chitin", "xenobiotic", "hive", "hniezdo", "tendon"],
            "FROST_GLACIAL_FRACTURE_CHASM": ["frost", "ľad", "ice", "glacial", "chasm", "trhlina", "snow", "blizzard", "avalanche"],
            "CALABI_YAU_QUANTUM_PORTAL": ["calabi", "yau", "quantum", "portal", "singularity", "manifold", "wormhole", "6d"]
        }

        for tid, kws in rules.items():
            for kw in kws:
                if kw in t_low:
                    scores[tid] += 2

        best_id = max(scores, key=scores.get)
        if scores[best_id] == 0:
            best_id = "CRYSTAL_AWAKENING_RITUAL"

        return self._presets[best_id]


# Global Singleton Instance
GLOBAL_PROCEDURAL_VISUAL_TEMPLATES = KrystalProceduralVisualTemplateSystem()
