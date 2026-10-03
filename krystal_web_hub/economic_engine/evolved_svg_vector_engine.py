"""
KRYSTAL-STACK // EVOLVED HIGH-DETAIL SVG VECTOR ENGINE
=============================================================================
Procedural generator for high-detail, resolution-independent Scalable Vector
Graphics (SVG). Replaces unprepared/raw 3D mesh wireframes with evolved,
publication-grade vector art, technical blueprints, and tribal heraldry for
Poslední Kmen (Crystal, Toxic, Druid, Studna Duší).

Features:
1. Multi-layered SVG architecture with radial & linear gradients, glow filters,
   and precision blueprint grid backgrounds.
2. Exact mathematical annotations:
   - H(x, z) = H_0 + A * [ (1 - w_ridge) * F_fBm + w_ridge * R_ridge ] * C_fault
   - Delta H_thermal = -K_talus * max(0, ||grad H|| - tan theta_c) (theta_c = 35 deg)
   - SDF(x) = min(||x - c||) - r
   - E = int (grad phi)^2 dv
3. Vectorized sphere-tracing raymarching paths, hit indicators, and surface normals.
4. Tribal Crests & Radar HUD:
   - Crystal: Faceted frost polygon spires
   - Toxic: Serpentine Voronoi canyon networks
   - Druid: Ancient Tree of Life & talus slopes
   - Studna Duší: Cosmic vortex stabilization rings
5. Strict adherence to the 6 Max HP Vital Invariant.
"""

import math
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

VITAL_MAX_HP: int = 6
GOLDEN_RATIO: float = 1.61803398875
INV_GOLDEN_RATIO: float = 0.61803398875


class EvolvedSvgVectorEngine:
    """Procedural generator for evolved high-detail SVG graphics and blueprints."""

    def __init__(self):
        self.vital_max_hp = VITAL_MAX_HP

    def generate_master_blueprint_svg(self, width: int = 1600, height: int = 900) -> str:
        """
        Generates the evolved, master high-detail technical blueprint SVG
        synthesizing the floating island of Poslední Kmen, the biomes,
        and continuous mathematics & SDF raymarching annotations.
        """
        cx = width // 2
        cy = height // 2

        svg = []
        svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%" style="background-color: #030611; font-family: \'JetBrains Mono\', monospace;">')

        # ── <defs>: Gradients & Glow Filters ──────────────────────────────────
        svg.append('''
        <defs>
          <!-- Blueprint Grid Pattern -->
          <pattern id="gridPattern" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(0, 180, 255, 0.07)" stroke-width="1"/>
            <path d="M 200 0 L 0 0 0 200" fill="none" stroke="rgba(0, 180, 255, 0.15)" stroke-width="1.5"/>
          </pattern>

          <!-- Glow Filters -->
          <filter id="glowCyan" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="6" result="blur"/>
            <feMerge>
              <feMergeNode in="blur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>

          <filter id="glowMagenta" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="8" result="blur"/>
            <feMerge>
              <feMergeNode in="blur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>

          <filter id="glowGreen" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="6" result="blur"/>
            <feMerge>
              <feMergeNode in="blur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>

          <!-- Radial Gradients -->
          <radialGradient id="gradVortex" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stop-color="#ffffff"/>
            <stop offset="25%" stop-color="#00f0ff"/>
            <stop offset="65%" stop-color="#bf5af2"/>
            <stop offset="100%" stop-color="rgba(191, 90, 242, 0)"/>
          </radialGradient>

          <linearGradient id="gradCrystal" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#ffffff" stop-opacity="0.9"/>
            <stop offset="40%" stop-color="#00f0ff" stop-opacity="0.7"/>
            <stop offset="100%" stop-color="#003366" stop-opacity="0.8"/>
          </linearGradient>

          <linearGradient id="gradToxic" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stop-color="#00ff88" stop-opacity="0.9"/>
            <stop offset="70%" stop-color="#104020" stop-opacity="0.8"/>
            <stop offset="100%" stop-color="#051508" stop-opacity="0.9"/>
          </linearGradient>

          <linearGradient id="gradDruid" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#fbbf24" stop-opacity="0.9"/>
            <stop offset="50%" stop-color="#b45309" stop-opacity="0.8"/>
            <stop offset="100%" stop-color="#451a03" stop-opacity="0.8"/>
          </linearGradient>
        </defs>
        ''')

        # ── Background Blueprint Grid ─────────────────────────────────────────
        svg.append(f'<rect width="{width}" height="{height}" fill="url(#gridPattern)"/>')

        # ── Outer Blueprint Border & Coordinates ──────────────────────────────
        svg.append(f'<rect x="20" y="20" width="{width - 40}" height="{height - 40}" fill="none" stroke="rgba(0, 180, 255, 0.4)" stroke-width="1.5"/>')
        svg.append(f'<rect x="25" y="25" width="{width - 50}" height="{height - 50}" fill="none" stroke="rgba(0, 180, 255, 0.15)" stroke-width="1" stroke-dasharray="4, 4"/>')

        # Calibration marks in corners
        for x_c in [35, width - 35]:
            for y_c in [35, height - 35]:
                svg.append(f'<circle cx="{x_c}" cy="{y_c}" r="8" fill="none" stroke="#00f0ff" stroke-width="1"/>')
                svg.append(f'<line x1="{x_c - 12}" y1="{y_c}" x2="{x_c + 12}" y2="{y_c}" stroke="#00f0ff" stroke-width="1"/>')
                svg.append(f'<line x1="{x_c}" y1="{y_c - 12}" x2="{x_c}" y2="{y_c + 12}" stroke="#00f0ff" stroke-width="1"/>')

        # ── Top-Left: Document Header ─────────────────────────────────────────
        svg.append(f'''
        <g transform="translate(50, 60)">
          <text x="0" y="0" fill="#00f0ff" font-size="18" font-weight="900" letter-spacing="2">PROJECT: POSLEDNÍ KMEN // CELESTIAL ISLAND (NB-093)</text>
          <text x="0" y="22" fill="#94a3b8" font-size="11">SCALE 1:5000 · TOPOLOGICAL MANIFOLD · MULTI-BACKEND ACCELERATION</text>
          <text x="0" y="38" fill="#f59e0b" font-size="11">VORTEX CORE: STUDNA DUŠÍ (λ = 515nm / 660nm) · INVARIANT: VITAL_MAX_HP = 6</text>
        </g>
        ''')

        # ── Top-Right: Tribal Seals ───────────────────────────────────────────
        svg.append(f'''
        <g transform="translate({width - 320}, 50)">
          <!-- Seal 1: Poslední Kmen -->
          <circle cx="30" cy="30" r="24" fill="rgba(0, 240, 255, 0.1)" stroke="#00f0ff" stroke-width="1.5"/>
          <polygon points="30,12 44,42 16,42" fill="none" stroke="#00f0ff" stroke-width="1.5"/>
          <text x="30" y="68" fill="#94a3b8" font-size="9" text-anchor="middle">POSLEDNÍ KMEN</text>

          <!-- Seal 2: Čtyři Směry -->
          <circle cx="110" cy="30" r="24" fill="rgba(245, 158, 11, 0.1)" stroke="#f59e0b" stroke-width="1.5"/>
          <circle cx="110" cy="30" r="16" fill="none" stroke="#f59e0b" stroke-width="1" stroke-dasharray="3, 3"/>
          <line x1="110" y1="12" x2="110" y2="48" stroke="#f59e0b" stroke-width="1.5"/>
          <line x1="92" y1="30" x2="128" y2="30" stroke="#f59e0b" stroke-width="1.5"/>
          <text x="110" y="68" fill="#94a3b8" font-size="9" text-anchor="middle">ČTYŘI SMĚRY</text>

          <!-- Seal 3: Strážci -->
          <circle cx="190" cy="30" r="24" fill="rgba(191, 90, 242, 0.1)" stroke="#bf5af2" stroke-width="1.5"/>
          <path d="M 190,14 A 16,16 0 1,1 189.9,14" fill="none" stroke="#bf5af2" stroke-width="2" stroke-dasharray="10, 4"/>
          <text x="190" y="68" fill="#94a3b8" font-size="9" text-anchor="middle">STRÁŽCI</text>
        </g>
        ''')

        # ── Center: Floating Celestial Island Biomes ──────────────────────────
        island_y = cy + 20

        # Island Underbelly & Rock Roots (Layered facets)
        svg.append(f'''
        <g id="islandUnderbelly" stroke="#00f0ff" stroke-width="0.8" opacity="0.85">
          <polygon points="{cx - 240},{island_y + 30} {cx},{island_y + 190} {cx - 80},{island_y + 80}" fill="#081020"/>
          <polygon points="{cx},{island_y + 190} {cx + 240},{island_y + 30} {cx + 90},{island_y + 80}" fill="#0a1528"/>
          <polygon points="{cx - 80},{island_y + 80} {cx},{island_y + 190} {cx + 90},{island_y + 80}" fill="#0c1a32"/>
          <!-- Root tendrils -->
          <path d="M {cx - 40},{island_y + 140} Q {cx - 80},{island_y + 220} {cx - 60},{island_y + 260}" fill="none" stroke="#f59e0b" stroke-width="1.8"/>
          <path d="M {cx + 20},{island_y + 160} Q {cx + 60},{island_y + 230} {cx + 40},{island_y + 270}" fill="none" stroke="#f59e0b" stroke-width="1.8"/>
        </g>
        ''')

        # Biome 1: Crystal (North-West / Top-Left) - Faceted Spikes
        svg.append(f'''
        <g id="biomeCrystal" filter="url(#glowCyan)">
          <!-- Main Ice Spire -->
          <polygon points="{cx - 180},{island_y - 140} {cx - 240},{island_y + 20} {cx - 150},{island_y + 10}" fill="url(#gradCrystal)" stroke="#00f0ff" stroke-width="1.5"/>
          <polygon points="{cx - 180},{island_y - 140} {cx - 150},{island_y + 10} {cx - 110},{island_y - 40}" fill="#ffffff" fill-opacity="0.8" stroke="#00f0ff" stroke-width="1.5"/>
          <polygon points="{cx - 110},{island_y - 40} {cx - 150},{island_y + 10} {cx - 90},{island_y + 20}" fill="#38bdf8" fill-opacity="0.7" stroke="#00f0ff" stroke-width="1.2"/>

          <!-- Secondary Crystal Peaks -->
          <polygon points="{cx - 260},{island_y - 60} {cx - 290},{island_y + 25} {cx - 220},{island_y + 20}" fill="url(#gradCrystal)" stroke="#00f0ff" stroke-width="1.2"/>
          <polygon points="{cx - 130},{island_y - 90} {cx - 160},{island_y - 10} {cx - 90},{island_y - 20}" fill="#7dd3fc" fill-opacity="0.6" stroke="#00f0ff" stroke-width="1.2"/>

          <text x="{cx - 260}" y="{island_y - 155}" fill="#00f0ff" font-size="12" font-weight="700">KŘIŠŤÁLOVÉ LEDOVÉ ŠTÍTY</text>
          <line x1="{cx - 190}" y1="{island_y - 145}" x2="{cx - 140}" y2="{island_y - 155}" stroke="#00f0ff" stroke-width="1"/>
          <text x="{cx - 260}" y="{island_y - 138}" fill="#94a3b8" font-size="9">FROST PEAKS · w_ridge = 0.88</text>
        </g>
        ''')

        # Biome 2: Toxic (South / Bottom) - Voronoi Canyons & Acid Streams
        svg.append(f'''
        <g id="biomeToxic" filter="url(#glowGreen)">
          <polygon points="{cx - 140},{island_y + 30} {cx - 30},{island_y + 120} {cx + 30},{island_y + 120} {cx + 140},{island_y + 30} {cx},{island_y + 30}" fill="url(#gradToxic)" stroke="#00ff88" stroke-width="1.5"/>
          <!-- Acid Rivers -->
          <path d="M {cx - 70},{island_y + 40} Q {cx - 20},{island_y + 70} {cx - 40},{island_y + 110}" fill="none" stroke="#00ff88" stroke-width="4"/>
          <path d="M {cx + 60},{island_y + 40} Q {cx + 10},{island_y + 75} {cx + 25},{island_y + 115}" fill="none" stroke="#00ff88" stroke-width="3"/>
          <circle cx="{cx - 10}" cy="{island_y + 90}" r="8" fill="#88ff00" opacity="0.8"/>
          <circle cx="{cx + 15}" cy="{island_y + 100}" r="5" fill="#88ff00" opacity="0.6"/>

          <text x="{cx - 60}" y="{island_y + 155}" fill="#00ff88" font-size="12" font-weight="700">TOXICKÉ KAŇONY JEDŮ</text>
          <text x="{cx - 60}" y="{island_y + 170}" fill="#94a3b8" font-size="9">VORONOI CANYONS · C_fault = 0.92</text>
        </g>
        ''')

        # Biome 3: Druid (East / Right) - World Tree Canopy & Talus Slopes
        svg.append(f'''
        <g id="biomeDruid">
          <!-- Talus Slope Base -->
          <polygon points="{cx + 50},{island_y + 20} {cx + 150},{island_y + 35} {cx + 260},{island_y - 20} {cx + 160},{island_y - 80}" fill="url(#gradDruid)" stroke="#f59e0b" stroke-width="1.5"/>
          
          <!-- Tree Canopies (Golden Foliage Circles) -->
          <circle cx="{cx + 170}" cy="{island_y - 80}" r="28" fill="#f59e0b" opacity="0.8" stroke="#fbbf24" stroke-width="1.5"/>
          <circle cx="{cx + 210}" cy="{island_y - 70}" r="24" fill="#d97706" opacity="0.8" stroke="#fbbf24" stroke-width="1.5"/>
          <circle cx="{cx + 190}" cy="{island_y - 110}" r="32" fill="#fbbf24" opacity="0.9" stroke="#fff" stroke-width="1"/>
          
          <!-- Tree Trunks -->
          <path d="M {cx + 188},{island_y - 80} L {cx + 188},{island_y - 30}" stroke="#78350f" stroke-width="6"/>
          
          <!-- Talus contour lines -->
          <path d="M {cx + 140},{island_y - 20} Q {cx + 190},{island_y} {cx + 240},{island_y - 10}" fill="none" stroke="#f59e0b" stroke-width="1" stroke-dasharray="3, 2"/>
          <path d="M {cx + 150},{island_y} Q {cx + 195},{island_y + 18} {cx + 230},{island_y + 10}" fill="none" stroke="#f59e0b" stroke-width="1" stroke-dasharray="3, 2"/>

          <text x="{cx + 150}" y="{island_y - 145}" fill="#f59e0b" font-size="12" font-weight="700">DRUIDSKÝ PODZIMNÍ LES</text>
          <text x="{cx + 150}" y="{island_y - 130}" fill="#94a3b8" font-size="9">TALUS SLOPES · θ_c ≈ 35°</text>
        </g>
        ''')

        # Biome 4: Studna Duší (Center Vortex Singularity)
        svg.append(f'''
        <g id="biomeStudnaDusi" filter="url(#glowMagenta)">
          <!-- Stabilization Rings -->
          <ellipse cx="{cx}" cy="{island_y - 10}" rx="110" ry="45" fill="none" stroke="#bf5af2" stroke-width="1.5" stroke-dasharray="6, 4"/>
          <ellipse cx="{cx}" cy="{island_y - 10}" rx="85" ry="34" fill="none" stroke="#00f0ff" stroke-width="1.2"/>
          <ellipse cx="{cx}" cy="{island_y - 10}" rx="60" ry="24" fill="url(#gradVortex)"/>

          <!-- Swirling energy spirals -->
          <path d="M {cx - 50},{island_y - 10} Q {cx},{island_y - 35} {cx + 35},{island_y - 15} T {cx},{island_y - 10}" fill="none" stroke="#ffffff" stroke-width="2"/>
          <path d="M {cx + 50},{island_y - 10} Q {cx},{island_y + 15} {cx - 35},{island_y - 5} T {cx},{island_y - 10}" fill="none" stroke="#00f0ff" stroke-width="2"/>

          <text x="{cx}" y="{island_y - 65}" fill="#bf5af2" font-size="13" font-weight="900" text-anchor="middle">STUDNA DUŠÍ</text>
          <text x="{cx}" y="{island_y - 50}" fill="#38bdf8" font-size="9" text-anchor="middle">STABILIZATION RING · SINGULARITY VORTEX</text>
        </g>
        ''')

        # ── SDF Raymarching Sphere-Tracing Vector Beam ─────────────────────────
        ray_x0 = cx - 360
        ray_y0 = island_y - 160
        hit_x = cx
        hit_y = island_y - 10

        svg.append(f'''
        <g id="sdfRaymarchVector">
          <!-- Ray Line -->
          <line x1="{ray_x0}" y1="{ray_y0}" x2="{hit_x}" y2="{hit_y}" stroke="#ff4757" stroke-width="2" stroke-dasharray="8, 4"/>
          <circle cx="{ray_x0}" cy="{ray_y0}" r="6" fill="#ff4757"/>
          <text x="{ray_x0 - 10}" y="{ray_y0 - 12}" fill="#ff4757" font-size="10" font-weight="700">RAY ORIGIN r₀</text>

          <!-- Decreasing Sphere Tracing Circles -->
          <circle cx="{ray_x0 + 90}" cy="{ray_y0 + 37}" r="34" fill="none" stroke="rgba(0, 255, 136, 0.4)" stroke-width="1.2"/>
          <circle cx="{ray_x0 + 190}" cy="{ray_y0 + 78}" r="22" fill="none" stroke="rgba(0, 255, 136, 0.45)" stroke-width="1.2"/>
          <circle cx="{ray_x0 + 270}" cy="{ray_y0 + 112}" r="14" fill="none" stroke="rgba(0, 255, 136, 0.5)" stroke-width="1.2"/>
          <circle cx="{ray_x0 + 325}" cy="{ray_y0 + 135}" r="7" fill="none" stroke="rgba(0, 255, 136, 0.6)" stroke-width="1.2"/>

          <!-- Surface Hit Spark -->
          <circle cx="{hit_x}" cy="{hit_y}" r="8" fill="#00ff88" filter="url(#glowGreen)"/>
          <text x="{hit_x + 15}" y="{hit_y - 5}" fill="#00ff88" font-size="10" font-weight="700">SURFACE HIT f(p) &lt; ε</text>

          <!-- Surface Normal Vector -->
          <line x1="{hit_x}" y1="{hit_y}" x2="{hit_x + 35}" y2="{hit_y - 50}" stroke="#00f0ff" stroke-width="2.5"/>
          <polygon points="{hit_x + 35},{hit_y - 50} {hit_x + 28},{hit_y - 42} {hit_x + 38},{hit_y - 40}" fill="#00f0ff"/>
          <text x="{hit_x + 42}" y="{hit_y - 45}" fill="#00f0ff" font-size="10">∇f(p)</text>
        </g>
        ''')

        # ── Bottom-Right: Exact Mathematical Formulas ────────────────────────
        svg.append(f'''
        <g transform="translate({width - 440}, {height - 230})">
          <rect width="390" height="175" fill="rgba(6, 12, 24, 0.9)" stroke="rgba(0, 180, 255, 0.4)" stroke-width="1" rx="6"/>
          
          <text x="20" y="28" fill="#00f0ff" font-size="12" font-weight="700">MATHEMATICAL FORMULATIONS</text>
          
          <!-- Energy Integral -->
          <text x="20" y="56" fill="#fff" font-size="14" font-family="serif">E = ∫ (∇φ)² dv</text>
          
          <!-- SDF Equation -->
          <text x="20" y="82" fill="#7dd3fc" font-size="13">SDF(x) = min(||x - c||) - r</text>
          
          <!-- Talus Weathering -->
          <text x="20" y="108" fill="#f59e0b" font-size="11">ΔH_thermal = -K_talus · max(0, ||∇H|| - tan θ_c)</text>
          
          <!-- Master Terrain -->
          <text x="20" y="130" fill="#94a3b8" font-size="10">H(x,z) = H₀ + A·[(1-w)F_fBm + w·R_ridge]·C_fault</text>

          <!-- NPU Cost Badge -->
          <text x="20" y="156" fill="#00ff88" font-size="11" font-weight="700">EVAL COST: 1 + 6 = 7 (CPU) → 1 + 1 = 2 (NPU TENSOR)</text>
        </g>
        ''')

        # ── Bottom-Left: Radar Chart & System Telemetry ───────────────────────
        svg.append(f'''
        <g transform="translate(60, {height - 250})">
          <!-- Radar Chart Axes -->
          <circle cx="90" cy="90" r="70" fill="none" stroke="rgba(0, 180, 255, 0.15)" stroke-width="1"/>
          <circle cx="90" cy="90" r="45" fill="none" stroke="rgba(0, 180, 255, 0.15)" stroke-width="1"/>
          <circle cx="90" cy="90" r="20" fill="none" stroke="rgba(0, 180, 255, 0.15)" stroke-width="1"/>
          
          <line x1="90" y1="10" x2="90" y2="170" stroke="rgba(0, 180, 255, 0.25)" stroke-width="1"/>
          <line x1="10" y1="90" x2="170" y2="90" stroke="rgba(0, 180, 255, 0.25)" stroke-width="1"/>

          <!-- Radar Polygon (Poslední Kmen Stats: Obtížnost, Kontrola, Poškození, Dosah) -->
          <polygon points="90,30 145,90 90,135 40,90" fill="rgba(0, 240, 255, 0.25)" stroke="#00f0ff" stroke-width="2"/>

          <!-- Labels -->
          <text x="90" y="6" fill="#94a3b8" font-size="9" text-anchor="middle">OBTÍŽNOST (4/6)</text>
          <text x="180" y="93" fill="#94a3b8" font-size="9">KONTROLA (5/6)</text>
          <text x="90" y="184" fill="#94a3b8" font-size="9" text-anchor="middle">POŠKOZENÍ (3/6)</text>
          <text x="2" y="93" fill="#94a3b8" font-size="9" text-anchor="end">DOSAH (4/6)</text>

          <!-- Driver Bypass Box -->
          <g transform="translate(200, 20)">
            <text x="0" y="15" fill="#00f0ff" font-size="11" font-weight="700">DRIVER BYPASS HARNESS</text>
            <text x="0" y="35" fill="#94a3b8" font-size="10">Intel Iris Xe (96 EUs / 672 threads)</text>
            <text x="0" y="52" fill="#ff4757" font-size="10">WDDM Stall: 185.0 μs</text>
            <text x="0" y="69" fill="#00ff88" font-size="10" font-weight="700">Krystal Bypass: 11.4 μs (16.2x)</text>
            <text x="0" y="86" fill="#f59e0b" font-size="10">NPU TOPS: 12.4 DirectML</text>
          </g>
        </g>
        ''')

        svg.append('</svg>')
        return "\n".join(svg)

    def generate_crystal_tribe_svg(self, size: int = 512) -> str:
        """High-detail SVG crest for Crystal Tribe (Vládci mrazu)."""
        c = size // 2
        return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" width="{size}" height="{size}" style="background:#020617;">
          <defs>
            <linearGradient id="crGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stop-color="#ffffff"/>
              <stop offset="50%" stop-color="#00f0ff"/>
              <stop offset="100%" stop-color="#0369a1"/>
            </linearGradient>
          </defs>
          <polygon points="{c},30 {size - 40},{size - 80} {c},{size - 130} {40},{size - 80}" fill="none" stroke="#00f0ff" stroke-width="2"/>
          <polygon points="{c},50 {c + 120},{c + 60} {c},{size - 100} {c - 120},{c + 60}" fill="url(#crGrad)" opacity="0.85" stroke="#ffffff" stroke-width="2"/>
          <line x1="{c}" y1="50" x2="{c}" y2="{size - 100}" stroke="#ffffff" stroke-width="2"/>
          <text x="{c}" y="{size - 25}" fill="#00f0ff" font-family="'JetBrains Mono', monospace" font-size="18" font-weight="900" text-anchor="middle">CRYSTAL // VLÁDCI MRAZU</text>
        </svg>'''

    def generate_toxic_tribe_svg(self, size: int = 512) -> str:
        """High-detail SVG crest for Toxic Tribe (Hnijící slatiny)."""
        c = size // 2
        return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" width="{size}" height="{size}" style="background:#020617;">
          <defs>
            <linearGradient id="txGrad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stop-color="#00ff88"/>
              <stop offset="100%" stop-color="#14532d"/>
            </linearGradient>
          </defs>
          <circle cx="{c}" cy="{c - 20}" r="170" fill="none" stroke="#00ff88" stroke-width="2" stroke-dasharray="8, 6"/>
          <path d="M {c - 80},{c - 90} Q {c},{c - 150} {c + 80},{c - 90} Q {c + 120},{c} {c},{c + 90} Q {c - 120},{c} {c - 80},{c - 90}" fill="url(#txGrad)" stroke="#00ff88" stroke-width="3"/>
          <circle cx="{c - 30}" cy="{c - 50}" r="10" fill="#000"/>
          <circle cx="{c + 30}" cy="{c - 50}" r="10" fill="#000"/>
          <text x="{c}" y="{size - 25}" fill="#00ff88" font-family="'JetBrains Mono', monospace" font-size="18" font-weight="900" text-anchor="middle">TOXIC // HNIJÍCÍ SLATINY</text>
        </svg>'''

    def generate_druid_tribe_svg(self, size: int = 512) -> str:
        """High-detail SVG crest for Druid Tribe (Pradávný les)."""
        c = size // 2
        return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" width="{size}" height="{size}" style="background:#020617;">
          <circle cx="{c}" cy="{c - 20}" r="170" fill="none" stroke="#f59e0b" stroke-width="2"/>
          <circle cx="{c}" cy="{c - 80}" r="65" fill="#f59e0b" opacity="0.85" stroke="#fbbf24" stroke-width="2"/>
          <circle cx="{c - 60}" cy="{c - 50}" r="50" fill="#d97706" opacity="0.8" stroke="#fbbf24" stroke-width="2"/>
          <circle cx="{c + 60}" cy="{c - 50}" r="50" fill="#d97706" opacity="0.8" stroke="#fbbf24" stroke-width="2"/>
          <path d="M {c},{c - 20} L {c},{c + 90}" stroke="#78350f" stroke-width="16"/>
          <text x="{c}" y="{size - 25}" fill="#f59e0b" font-family="'JetBrains Mono', monospace" font-size="18" font-weight="900" text-anchor="middle">DRUID // PRADÁVNÝ LES</text>
        </svg>'''

    def generate_studna_dusi_svg(self, size: int = 512) -> str:
        """High-detail SVG crest for Studna Duší (Central Vortex)."""
        c = size // 2
        return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" width="{size}" height="{size}" style="background:#020617;">
          <ellipse cx="{c}" cy="{c - 20}" rx="170" ry="80" fill="none" stroke="#bf5af2" stroke-width="3" stroke-dasharray="10, 6"/>
          <ellipse cx="{c}" cy="{c - 20}" rx="120" ry="55" fill="none" stroke="#00f0ff" stroke-width="2"/>
          <ellipse cx="{c}" cy="{c - 20}" rx="70" ry="30" fill="#bf5af2" opacity="0.85"/>
          <circle cx="{c}" cy="{c - 20}" r="22" fill="#ffffff"/>
          <text x="{c}" y="{size - 25}" fill="#bf5af2" font-family="'JetBrains Mono', monospace" font-size="18" font-weight="900" text-anchor="middle">STUDNA DUŠÍ // METAVORTEX</text>
        </svg>'''


GLOBAL_EVOLVED_SVG_ENGINE = EvolvedSvgVectorEngine()
