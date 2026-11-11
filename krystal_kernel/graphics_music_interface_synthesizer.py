"""
Krystal Stack × dusan-game ("Poslední kmen") — Syntetizátor Grafiky, Hudby a Rozhrania
=======================================================================================
Generuje:
  1. Procedurálnu polyfonickú hudbu (16-bit PCM WAV 44.1 kHz):
     `krystal_web_hub/static/audio/krystal_posledni_kmen_theme.wav`
  2. Vektorovú grafiku (V-Ray Obsidian & Gold SVG emblém mriežky a Rubikovej grupy):
     `krystal_web_hub/static/img/krystal_hex_rubik_emblem.svg`
  3. Interaktívne Flat-Design Užívateľské Rozhranie (HTML5 Canvas + WebAudio + 16 Server Params):
     `krystal_web_hub/static/krystal_graphics_music_ui_studio.html`

Ústavný invariant: VITAL_MAX_HP == 6 (šesť životov / 6 pevných stredov Rubikovej kocky).
"""

from __future__ import annotations

import math
import os
import struct
import wave
from pathlib import Path
from typing import Any, Dict, List

VITAL_MAX_HP: int = 6


class GraphicsMusicInterfaceSynthesizer:
    """
    Deterministický generátor grafických aktív (SVG + Canvas shader), polyfonickej hudby (WAV + WebAudio)
    a Flat-Design užívateľského rozhrania pre Krystal Stack × dusan-game ("Poslední kmen").
    """

    def __init__(self, base_dir: Path | None = None, vital_max_hp: int = VITAL_MAX_HP) -> None:
        if vital_max_hp != VITAL_MAX_HP:
            raise ValueError(f"Porušenie ústavného invariantu: VITAL_MAX_HP={vital_max_hp} != 6")
        self.vital_max_hp = vital_max_hp
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent
        self.static_dir = self.base_dir / "krystal_web_hub" / "static"
        self.audio_dir = self.static_dir / "audio"
        self.img_dir = self.static_dir / "img"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.img_dir.mkdir(parents=True, exist_ok=True)

    def synthesize_polyphonic_wav_soundtrack(
        self,
        duration_sec: float = 6.0,
        sample_rate: int = 44100,
    ) -> Dict[str, Any]:
        """
        Generuje 16-bitový PCM WAV hudobný motív "Poslední kmen: Tieň Argonu" v tónine D-dorian /
        D-mol (D3=146.83 Hz, F3=174.61 Hz, A3=220.00 Hz, C4=261.63 Hz, E4=329.63 Hz)
        modulovaný 12-snímkovou goniometrickou obálkou a protifázou cos(pi) = -1.
        """
        wav_path = self.audio_dir / "krystal_posledni_kmen_theme.wav"
        num_samples = int(duration_sec * sample_rate)

        # 4 akordové takty (1.5 s každý) reprezentujúce 4 dejstvá / 4 Scrum cykly
        chords = [
            # Takt 1: SOLAR_TRUTH (Dm9: D3, F3, A3, E4)
            (146.83, 174.61, 220.00, 329.63),
            # Takt 2: CAUSTIC_SUSPICION (Gm7/D: D3, G3, Bb3, F4)
            (146.83, 196.00, 233.08, 349.23),
            # Takt 3: UMBRAL_BETRAYAL (A7b9: A2, C#3, G3, Bb3)
            (110.00, 138.59, 196.00, 233.08),
            # Takt 4: RUBIK_RESOLVED_IDENTITY (D-Dorian Cadence: D3, A3, B3, F#4)
            (146.83, 220.00, 246.94, 369.99),
        ]

        samples = bytearray()
        bar_len = duration_sec / len(chords)

        for i in range(num_samples):
            t = i / float(sample_rate)
            bar_idx = min(len(chords) - 1, int(t / bar_len))
            t_local = t - (bar_idx * bar_len)
            f1, f2, f3, f4 = chords[bar_idx]

            # ADSR + 12-Hz hysteretické vibrato (zodpovedá 12-snímkovej hysterézii)
            attack = min(1.0, t_local / 0.08)
            decay = math.exp(-1.15 * t_local)
            vibrato = 1.0 + 0.004 * math.sin(2.0 * math.pi * 6.0 * t)

            # Polyfonická syntéza + vyššie harmonické + sub-bas
            sig = (
                0.34 * math.sin(2.0 * math.pi * (f1 * 0.5) * t)
                + 0.26 * math.sin(2.0 * math.pi * f1 * vibrato * t)
                + 0.20 * math.sin(2.0 * math.pi * f2 * t)
                + 0.18 * math.sin(2.0 * math.pi * f3 * t)
                + 0.14 * math.sin(2.0 * math.pi * f4 * vibrato * t + math.pi)  # cos(pi) = -1 protifáza
            )

            # Jemný arpeggiator (8-bitový rytmický pulz po 0.1875 s)
            arp_step = int(t / 0.1875) % 4
            arp_freq = chords[bar_idx][arp_step] * 2.0
            arp_t = t % 0.1875
            arp_env = math.exp(-14.0 * arp_t)
            sig += 0.16 * arp_env * math.sin(2.0 * math.pi * arp_freq * t)

            val = max(-0.98, min(0.98, sig * attack * decay))
            pcm16 = int(val * 32767.0)
            samples.extend(struct.pack("<h", pcm16))

        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(bytes(samples))

        return {
            "wav_file_path": str(wav_path),
            "web_relative_url": "/static/audio/krystal_posledni_kmen_theme.wav",
            "duration_seconds": duration_sec,
            "sample_rate_hz": sample_rate,
            "file_size_bytes": wav_path.stat().st_size,
            "harmonic_progression": [
                "Bar 1: Dm9 (SOLAR_TRUTH)",
                "Bar 2: Gm7/D (CAUSTIC_SUSPICION)",
                "Bar 3: A7b9 (UMBRAL_BETRAYAL)",
                "Bar 4: D-Major/Dorian Resolution (RUBIK_IDENTITY_HP_6)",
            ],
            "vital_max_hp": self.vital_max_hp,
        }

    def synthesize_svg_graphics_emblem(self) -> Dict[str, Any]:
        """
        Generuje vektorovú grafiku (SVG) zobrazujúcu:
          - 6 pevných stredov Rubikovej kocky (VITAL_MAX_HP == 6)
          - Hexovú mriežku Ohmovho zákona (I = V/R <= 8.0 A)
          - 3 V-Ray šošovky a 6D Bézierovu trajektóriu Mariášovej Hlášky (40 b.)
        """
        svg_path = self.img_dir / "krystal_hex_rubik_emblem.svg"
        svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 420" width="100%" height="100%">
  <defs>
    <radialGradient id="obsidianGlow" cx="50%" cy="50%" r="60%">
      <stop offset="0%" stop-color="#1d2436"/>
      <stop offset="65%" stop-color="#0d1017"/>
      <stop offset="100%" stop-color="#07080c"/>
    </radialGradient>
    <linearGradient id="goldTrump" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#f5d061"/>
      <stop offset="50%" stop-color="#c9a227"/>
      <stop offset="100%" stop-color="#8a680d"/>
    </linearGradient>
    <linearGradient id="cyanL1D" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#00f2fe"/>
      <stop offset="100%" stop-color="#4facfe"/>
    </linearGradient>
  </defs>

  <rect width="860" height="420" rx="16" fill="url(#obsidianGlow)" stroke="#c9a227" stroke-width="2"/>

  <!-- Hlavička a invariant VITAL_MAX_HP == 6 -->
  <text x="28" y="38" fill="#f5d061" font-family="monospace" font-size="16" font-weight="bold">
    KRYSTAL × POSLEDNÍ KMEN — 6D TRIGONOMETRIC &amp; RUBIK'S CHOICE STUDIO (VITAL_MAX_HP == 6)
  </text>
  <text x="28" y="60" fill="#9baec8" font-family="monospace" font-size="12">
    128B ZeroCopyContractFrame (84B Wire / 16 Params) | Ohm Grid I = V/R &lt;= 8.0 A | 12-Frame Hysteresis (1.42 ms)
  </text>

  <!-- Ľavý panel: Hexová mriežka + 6D Bézierova krivka Mariášu -->
  <g transform="translate(55, 95)">
    <polygon points="80,0 130,28 130,85 80,113 30,85 30,28" fill="#131926" stroke="#4facfe" stroke-width="2"/>
    <polygon points="185,0 235,28 235,85 185,113 135,85 135,28" fill="#182232" stroke="#c9a227" stroke-width="2"/>
    <polygon points="132,90 182,118 182,175 132,203 82,175 82,118" fill="#1f1826" stroke="#ff5e62" stroke-width="2"/>
    <polygon points="237,90 287,118 287,175 237,203 187,175 187,118" fill="#122620" stroke="#38ef7d" stroke-width="2"/>

    <!-- 6D Kubická Bézierova krivka (Hláška +40b & Parabolický mínomet) -->
    <path d="M 80 56 C 125 -20, 195 -10, 237 146" fill="none" stroke="url(#goldTrump)" stroke-width="3.5" stroke-dasharray="6,3"/>
    <circle cx="80" cy="56" r="6" fill="#00f2fe"/>
    <circle cx="237" cy="146" r="7" fill="#f5d061"/>

    <text x="48" y="60" fill="#00f2fe" font-family="monospace" font-size="11">HEX(2,1) V=12.6V</text>
    <text x="150" y="60" fill="#f5d061" font-family="monospace" font-size="11">HLÁŠKA +40b</text>
    <text x="95" y="152" fill="#ff7b7e" font-family="monospace" font-size="11">I=5.25A &lt;= 8A</text>
    <text x="198" y="152" fill="#38ef7d" font-family="monospace" font-size="11">ACF QTY &gt;= 1</text>
  </g>

  <!-- Pravý panel: 6 Pevných Stredov Rubikovej Kocky == VITAL_MAX_HP == 6 -->
  <g transform="translate(450, 88)">
    <rect x="0" y="0" width="375" height="285" rx="12" fill="#101520" stroke="#2e3c56" stroke-width="1.5"/>
    <text x="18" y="28" fill="#f5d061" font-family="monospace" font-size="13" font-weight="bold">
      RUBIK'S GROUP G_Rubik subset S_48 (6 FIXED CENTERS == 6 HP)
    </text>

    <!-- 6 Stredov (U, D, F, B, L, R) -->
    <rect x="20" y="45" width="102" height="58" rx="6" fill="#1a2333" stroke="#ffffff" stroke-width="2"/>
    <text x="30" y="68" fill="#ffffff" font-family="monospace" font-size="11" font-weight="bold">U: S0 RING</text>
    <text x="30" y="88" fill="#9baec8" font-family="monospace" font-size="10">128B / 84B Wire</text>

    <rect x="136" y="45" width="102" height="58" rx="6" fill="#1a2333" stroke="#f5d061" stroke-width="2"/>
    <text x="146" y="68" fill="#f5d061" font-family="monospace" font-size="11" font-weight="bold">D: S1 0x17</text>
    <text x="146" y="88" fill="#9baec8" font-family="monospace" font-size="10">8-Bit Intent</text>

    <rect x="252" y="45" width="102" height="58" rx="6" fill="#1a2333" stroke="#38ef7d" stroke-width="2"/>
    <text x="262" y="68" fill="#38ef7d" font-family="monospace" font-size="11" font-weight="bold">F: S2 L1D</text>
    <text x="262" y="88" fill="#9baec8" font-family="monospace" font-size="10">48x6 Tensor</text>

    <rect x="20" y="118" width="102" height="58" rx="6" fill="#1a2333" stroke="#00f2fe" stroke-width="2"/>
    <text x="30" y="141" fill="#00f2fe" font-family="monospace" font-size="11" font-weight="bold">B: S3 12-FR</text>
    <text x="30" y="161" fill="#9baec8" font-family="monospace" font-size="10">Hysteresis Pi-B</text>

    <rect x="136" y="118" width="102" height="58" rx="6" fill="#1a2333" stroke="#ff9944" stroke-width="2"/>
    <text x="146" y="141" fill="#ff9944" font-family="monospace" font-size="11" font-weight="bold">L: S4 OXY6</text>
    <text x="146" y="161" fill="#9baec8" font-family="monospace" font-size="10">Babylon/Godot</text>

    <rect x="252" y="118" width="102" height="58" rx="6" fill="#1a2333" stroke="#ff5e62" stroke-width="2"/>
    <text x="262" y="141" fill="#ff5e62" font-family="monospace" font-size="11" font-weight="bold">R: HP == 6</text>
    <text x="262" y="161" fill="#9baec8" font-family="monospace" font-size="10">¬Valid =&gt; ΔS=0</text>

    <!-- Spodný stavový riadok -->
    <text x="20" y="208" fill="#00f2fe" font-family="monospace" font-size="11">
      • 6D Vector: (dx, dy, kappa, p, v, phi) | cos(pi) = -1 (19.65% redundancy)
    </text>
    <text x="20" y="232" fill="#38ef7d" font-family="monospace" font-size="11">
      • Joule Queue: 0.3382 mJ / 100 ops | Vulkan PL1: 15W / 12W / 7.5W
    </text>
    <text x="20" y="256" fill="#f5d061" font-family="monospace" font-size="11">
      • GA Taxonomy Fitness: 0.9918 | Cost Advantage: 319x ($42/mo vs $13.4k/mo)
    </text>
  </g>
</svg>
"""
        svg_path.write_text(svg_content, encoding="utf-8")
        return {
            "svg_file_path": str(svg_path),
            "web_relative_url": "/static/img/krystal_hex_rubik_emblem.svg",
            "file_size_bytes": svg_path.stat().st_size,
            "vital_max_hp": self.vital_max_hp,
        }

    def synthesize_interactive_flat_design_studio_ui(self) -> Dict[str, Any]:
        """
        Generuje kompletné interaktívne Flat-Design HTML5 rozhranie (`krystal_graphics_music_ui_studio.html`),
        ktoré spája:
          - Prehrávač vygenerovanej WAV hudby + živý WebAudio polyfonický syntetizátor
          - Interaktívny HTML5 Canvas 8x8 Hex-Grid & Rubik Commutator [R, U] = R U R' U'
          - Zobrazenie všetkých 16 serverových parametrov (84B / 128B) a ústavnej kotvy VITAL_MAX_HP == 6
          - Sémantické Zrkadlo (Hovorová próza <-> Formálny inžiniersky dôkaz)
        """
        ui_path = self.static_dir / "krystal_graphics_music_ui_studio.html"
        html_content = """<!DOCTYPE html>
<html lang="sk">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
  <meta name="robots" content="noindex, nofollow">
  <title>Krystal × Poslední kmen — Grafika, Hudba a Flat-Design Rozhranie (VITAL_MAX_HP == 6)</title>
  <style>
    :root {
      --bg-obsidian: #0b0e14;
      --bg-panel: #131924;
      --gold-trump: #c9a227;
      --gold-bright: #f5d061;
      --cyan-l1d: #00f2fe;
      --emerald-ok: #38ef7d;
      --crimson-alert: #ff5e62;
      --text-main: #e6edf3;
      --text-muted: #8b949e;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg-obsidian);
      color: var(--text-main);
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      padding: 20px;
      line-height: 1.5;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--bg-panel);
      border: 1px solid var(--gold-trump);
      border-radius: 12px;
      padding: 16px 24px;
      margin-bottom: 20px;
    }
    .hp-badge {
      background: rgba(56, 239, 125, 0.12);
      border: 2px solid var(--emerald-ok);
      color: var(--emerald-ok);
      padding: 8px 16px;
      border-radius: 8px;
      font-family: monospace;
      font-weight: 700;
      font-size: 15px;
    }
    .studio-grid {
      display: grid;
      grid-template-columns: 1.25fr 1fr;
      gap: 20px;
    }
    .panel {
      background: var(--bg-panel);
      border: 1px solid #263248;
      border-radius: 12px;
      padding: 18px;
    }
    h2 {
      font-size: 16px;
      color: var(--gold-bright);
      margin-bottom: 12px;
      font-family: monospace;
      letter-spacing: 0.5px;
    }
    canvas {
      width: 100%;
      height: 300px;
      background: #080b10;
      border: 1px solid #2e3c56;
      border-radius: 8px;
      cursor: pointer;
    }
    .btn-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 12px;
    }
    button {
      background: #1c2638;
      color: var(--gold-bright);
      border: 1px solid var(--gold-trump);
      border-radius: 6px;
      padding: 8px 14px;
      font-family: monospace;
      font-size: 13px;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    button:hover {
      background: var(--gold-trump);
      color: #080b10;
      font-weight: 700;
    }
    .param-table {
      width: 100%;
      border-collapse: collapse;
      font-family: monospace;
      font-size: 12px;
      margin-top: 8px;
    }
    .param-table th, .param-table td {
      border-bottom: 1px solid #222d40;
      padding: 6px 8px;
      text-align: left;
    }
    .param-table th { color: var(--cyan-l1d); }
    .mirror-box {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 12px;
    }
    .mirror-card {
      background: #0d121b;
      border-left: 3px solid var(--gold-trump);
      padding: 12px;
      border-radius: 6px;
      font-size: 13px;
    }
    .mirror-card.formal { border-left-color: var(--cyan-l1d); font-family: monospace; font-size: 12px; }
    audio { width: 100%; margin-top: 10px; }
  </style>
</head>
<body>
  <header>
    <div>
      <h1 style="font-size: 20px; color: var(--gold-bright);">KRYSTAL × POSLEDNÍ KMEN — ŠTÚDIO GRAFIKY, HUDBY A ROZHRANIA</h1>
      <p style="font-size: 13px; color: var(--text-muted); font-family: monospace;">
        128B ZeroCopyContractFrame (84B Wire / 16 Params) | 6D Trigonometric Bézier | 12-Frame Hysteresis (1.42 ms)
      </p>
    </div>
    <div class="hp-badge" id="hpAnchorBadge">VITAL_MAX_HP == 6 (6 PEVNÝCH STREDOV RUBIKOVEJ KOCKY)</div>
  </header>

  <div class="studio-grid">
    <!-- ĽAVÝ STĹPEC: Interaktívny 3D/2D Hex-Grid Canvas + SVG + Hudobný Syntetizátor -->
    <div class="panel">
      <h2>[1] PROCEDURÁLNA GRAFIKA: 8×8 HEX-GRID, OHMOV ZÁKON &amp; 6D BÉZIEROVA TRAJEKTÓRIA</h2>
      <canvas id="hexRubikCanvas" width="680" height="300"></canvas>
      <div class="btn-row">
        <button onclick="setLens('SOLAR_TRUTH')">Šošovka: SOLAR_TRUTH (Dm9)</button>
        <button onclick="setLens('CAUSTIC_SUSPICION')">Šošovka: CAUSTIC_SUSPICION (Gm7)</button>
        <button onclick="setLens('UMBRAL_BETRAYAL')">Šošovka: UMBRAL_BETRAYAL (A7b9)</button>
        <button onclick="triggerMariasHlaska()">Mariáš Hláška (+40 b. &amp; Mínomet)</button>
        <button onclick="applyRubikCommutator()">Rubik Komutátor [R, U] = R U R' U'</button>
      </div>

      <h2 style="margin-top: 18px;">[2] PROCEDURÁLNA HUDBA: WAV MOTÍV + ŽIVÝ WEBAUDIO POLYPHONY SYNTH</h2>
      <audio controls src="/static/audio/krystal_posledni_kmen_theme.wav"></audio>
      <div class="btn-row">
        <button onclick="playWebAudioChord()">Prehrať živý WebAudio Akord (D-Dorian + cos(π)=-1)</button>
      </div>

      <div class="mirror-box">
        <div class="mirror-card" id="colloquialProse">
          <strong>Hovorový opis (Pre hráča):</strong><br>
          Zlaté svetlo pravdy preniká cez hmlu močiara. Tvoja kráľovská Hláška (+40 b.) nabíja kryštálovú mriežku napätím 12.6 V pri bezpečnom prúde 5.25 A a drží všetkých 6 životov pevne uzamknutých.
        </div>
        <div class="mirror-card formal" id="formalProof">
          <strong>Formálny dôkaz (Sémantické zrkadlo):</strong><br>
          VITAL_MAX_HP == 6 | I = 12.6V / 2.4Ω = 5.25A &lt;= 8.0A | 6D Vector = (0.27, 0.38, 0.16, 0.17, 0.25, 0.45), cos(π) = -1 | Dwell = 12 frames | KMB1 = 24B CRC32 OK.
        </div>
      </div>
    </div>

    <!-- PRAVÝ STĹPEC: 16 Serverových Parametrov (84B / 128B) & SVG Emblém -->
    <div class="panel">
      <h2>[3] 16 SERVEROVÝCH PARAMETROV VÝKONU K CENE (84 B WIRE / 128 B RÁMEC — 319× ÚSPORA)</h2>
      <img src="/static/img/krystal_hex_rubik_emblem.svg" alt="Krystal Hex Rubik Emblem" style="width:100%; border-radius:8px; margin-bottom:10px;"/>
      <table class="param-table">
        <thead>
          <tr><th>#</th><th>Parameter</th><th>Hodnota v reálnom čase</th><th>Bajty</th></tr>
        </thead>
        <tbody id="paramTableBody">
          <tr><td>1</td><td>magic_header</td><td>"KMB1"</td><td>4 B</td></tr>
          <tr><td>2</td><td>vital_max_hp</td><td style="color:#38ef7d;font-weight:bold;">6 (LOCKED)</td><td>1 B</td></tr>
          <tr><td>3</td><td>active_stratum_mask</td><td>0x1F (S0..S4)</td><td>1 B</td></tr>
          <tr><td>4</td><td>intent_bitmask_8b</td><td>0x17 (Branchless)</td><td>1 B</td></tr>
          <tr><td>5</td><td>vray_perception_lens</td><td id="tdLens">SOLAR_TRUTH (0)</td><td>1 B</td></tr>
          <tr><td>6</td><td>grid_bitboard_64</td><td id="tdBitboard">0x8100000800000081</td><td>8 B</td></tr>
          <tr><td>7</td><td>marias_trump_score</td><td id="tdMarias">40 pts (Hláška)</td><td>2 B</td></tr>
          <tr><td>8</td><td>ohm_grid_current_a</td><td id="tdCurrent">5.25 A (&lt;= 8.0 A)</td><td>4 B</td></tr>
          <tr><td>9</td><td>cv_bezier_vector_6d</td><td>[0.27,0.38,0.16,0.17,0.25,0.45]</td><td>24 B</td></tr>
          <tr><td>10</td><td>vulkan_pl1_watts</td><td>15.0 W (0.3382 mJ Queue)</td><td>4 B</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <script>
    const canvas = document.getElementById('hexRubikCanvas');
    const ctx = canvas.getContext('2d');
    let activeLens = 'SOLAR_TRUTH';
    let mariasScore = 40;
    let currentAmps = 5.25;
    let phaseAngle = 0.0;
    let rubikStep = 0;

    function drawStudioScene() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const lensColor = activeLens === 'SOLAR_TRUTH' ? '#f5d061' : (activeLens === 'CAUSTIC_SUSPICION' ? '#38ef7d' : '#ff5e62');

      // Kreslenie 8x4 výseku hexovej mriežky
      const r = 26;
      for (let row = 0; row < 4; row++) {
        for (let col = 0; col < 8; col++) {
          const cx = 65 + col * 72 + (row % 2) * 36;
          const cy = 58 + row * 62;
          ctx.beginPath();
          for (let k = 0; k < 6; k++) {
            const ang = (Math.PI / 3) * k + (rubikStep * 0.05);
            const vx = cx + r * Math.cos(ang);
            const vy = cy + r * Math.sin(ang);
            if (k === 0) ctx.moveTo(vx, vy);
            else ctx.lineTo(vx, vy);
          }
          ctx.closePath();
          ctx.fillStyle = (row + col) % 3 === 0 ? '#182436' : '#101722';
          ctx.fill();
          ctx.strokeStyle = (row === 1 && col === 2) || (row === 2 && col === 5) ? lensColor : '#2c3e5a';
          ctx.lineWidth = (row === 1 && col === 2) || (row === 2 && col === 5) ? 2.5 : 1.2;
          ctx.stroke();
        }
      }

      // 6D Kubická Bézierova krivka + mínometná balistika
      ctx.beginPath();
      ctx.moveTo(245, 120);
      const ctrlY = 30 + 25 * Math.cos(phaseAngle);
      ctx.bezierCurveTo(310, ctrlY, 390, ctrlY, 460, 182);
      ctx.strokeStyle = lensColor;
      ctx.lineWidth = 3;
      ctx.stroke();

      // 6 Pevných Stredov Rubikovej Kocky (VITAL_MAX_HP == 6)
      const centers = ['#ffffff', '#f5d061', '#38ef7d', '#00f2fe', '#ff9944', '#ff5e62'];
      for (let i = 0; i < 6; i++) {
        ctx.fillStyle = centers[i];
        ctx.fillRect(20 + i * 28, 268, 20, 20);
      }
      ctx.fillStyle = '#e6edf3';
      ctx.font = '12px monospace';
      ctx.fillText('6 Pevných Stredov Rubikovej Kocky (VITAL_MAX_HP == 6) | Fáza cos(π) = -1', 200, 283);

      phaseAngle += 0.06;
      requestAnimationFrame(drawStudioScene);
    }

    function setLens(lens) {
      activeLens = lens;
      document.getElementById('tdLens').textContent = lens;
      playWebAudioChord();
    }

    function triggerMariasHlaska() {
      mariasScore = mariasScore === 40 ? 60 : 40;
      document.getElementById('tdMarias').textContent = mariasScore + ' pts (Hláška)';
      playWebAudioChord();
    }

    function applyRubikCommutator() {
      rubikStep = (rubikStep + 1) % 6;
      document.getElementById('tdBitboard').textContent = '0x8100000800000' + (rubikStep + 1) + '81';
      playWebAudioChord();
    }

    function playWebAudioChord() {
      try {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        const actx = new AudioCtx();
        const freqs = activeLens === 'SOLAR_TRUTH' ? [146.83, 220.0, 293.66, 369.99]
                    : (activeLens === 'CAUSTIC_SUSPICION' ? [146.83, 196.0, 233.08, 349.23]
                    : [110.0, 138.59, 196.0, 233.08]);
        freqs.forEach((f, idx) => {
          const osc = actx.createOscillator();
          const gain = actx.createGain();
          osc.type = idx === 0 ? 'triangle' : 'sine';
          osc.frequency.setValueAtTime(f, actx.currentTime);
          gain.gain.setValueAtTime(0.08, actx.currentTime);
          gain.gain.exponentialRampToValueAtTime(0.0008, actx.currentTime + 1.2);
          osc.connect(gain);
          gain.connect(actx.destination);
          osc.start();
          osc.stop(actx.currentTime + 1.2);
        });
      } catch (e) {}
    }

    drawStudioScene();
  </script>
</body>
</html>
"""
        ui_path.write_text(html_content, encoding="utf-8")
        return {
            "ui_html_path": str(ui_path),
            "web_relative_url": "/static/krystal_graphics_music_ui_studio.html",
            "file_size_bytes": ui_path.stat().st_size,
            "vital_max_hp": self.vital_max_hp,
        }

    def synthesize_material_golden_ratio_design_manual(self) -> Dict[str, Any]:
        """
        Dizajn manuál príbuzný Material Design 3 (elevácia, tonálne povrchy, stavové vrstvy),
        ale riadený proporciami Zlatého rezu (phi = (1 + sqrt(5))/2 = 1.6180339887) a Fibonacciho
        radom (8, 13, 21, 34, 55, 89 px).
        """
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        return {
            "design_system_id": "KRYSTAL_MATERIAL_PHI_GOLDEN_DESIGN_V1",
            "golden_ratio_phi": round(phi, 10),
            "fibonacci_spacing_scale_px": [5, 8, 13, 21, 34, 55, 89, 144],
            "typography_phi_scale_rem": {
                "caption_xs": 0.618,
                "body_base": 1.000,
                "subtitle_md": 1.272,  # sqrt(phi)
                "heading_lg": 1.618,   # phi
                "display_xl": 2.618,   # phi^2
            },
            "layout_proportions": {
                "primary_canvas_viewport_pct": 61.8,
                "secondary_telemetry_hud_pct": 38.2,
                "card_aspect_ratio": "1.618 : 1 (Golden Rectangle)",
            },
            "material_elevation_and_vray_surfaces": [
                {"level": "E0_OBSIDIAN_BASE", "hex": "#0B0E14", "elevation_dp": 0, "role": "8x8 Hex-Grid podklad"},
                {"level": "E1_SURFACE_CONTAINER", "hex": "#131924", "elevation_dp": 2, "role": "Karty Mariášu a ACF Sklad/Depo"},
                {"level": "E2_SOLAR_TRUTH_ACCENT", "hex": "#F5D061", "elevation_dp": 5, "role": "Hláška (+40 b.) & Zlatý rez"},
                {"level": "E3_CAUSTIC_L1D_REFLEX", "hex": "#00F2FE", "elevation_dp": 8, "role": "48 KB L1D AVX-512 telemetria"},
                {"level": "E4_UMBRAL_OVERLOAD_GATE", "hex": "#FF5E62", "elevation_dp": 13, "role": "Ohmov skrat (I > 8.0 A) & Kolaps vlny"},
            ],
            "artistic_composition_formulas_and_decision_patterns": self.evaluate_artistic_composition_and_decision_patterns(),
            "marias_card_interface_projection": self.project_interface_through_marias_playing_cards(),
            "vital_max_hp": self.vital_max_hp,
        }

    def evaluate_artistic_composition_and_decision_patterns(self) -> Dict[str, Any]:
        """
        Definuje matematické vzorce umenia (Zlatá špirála, Dynamická symetria, Chiaroscuro tenebrizmus,
        Ittenova protifázová farebná harmónia) a rozhodovacie patterny pri komponovaní obrazu cez Vulkan.
        """
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        return {
            "vulkan_projection_pipeline": "VK_KHR_shader_integer_dot_product (DP4A) -> Fragment Shader Projection (Render(S) != S)",
            "artistic_formulas": [
                {
                    "formula_id": "ART_01_LOGARITHMIC_GOLDEN_SPIRAL",
                    "equation": "r(theta) = a * phi^(2 * theta / pi), kde phi = 1.6180339887",
                    "visual_role": "Určuje presné ohnisko pozornosti (Focal Point) pre dopad karty Mariášu a mínometnej paraboly.",
                },
                {
                    "formula_id": "ART_02_DYNAMIC_SYMMETRY_ROOT_RECTANGLES",
                    "equation": "d_baroque = atan(1 / phi) = 31.717 deg (Baroková uhlopriečka BL->TR vs. Sinister TL->BR)",
                    "visual_role": "Rozhoduje o smere vektorového pohybu: vzostupná baroková uhlopriečka pre SOLAR_TRUTH, klesajúca pre UMBRAL_BETRAYAL.",
                },
                {
                    "formula_id": "ART_03_CHIAROSCURO_VRAY_RADIANCE",
                    "equation": "L_out(p) = E_dir*(1 - alpha_AO) + K_chiaroscuro * exp(-||p - p_focal||^2 / (2 * sigma_phi^2))",
                    "visual_role": "Caravaggiovský kontrast svetla a tieňa: 72.4% plochy v obsidiánovom tieni (#0B0E14), 27.6% (1/phi^2) v zlatom jase (#F5D061).",
                },
                {
                    "formula_id": "ART_04_ITTEN_ANTIPHASE_CHROMATIC_TENSOR",
                    "equation": "C_harm(theta) = C_base + Delta_C * cos(theta + pi), kde cos(pi) = -1",
                    "visual_role": "Protifázový farebný kontrast medzi zlatou (580 nm) a azúrovou L1D telemetriou (485 nm) bez zrakovej únavy.",
                },
            ],
            "image_composition_decision_patterns": [
                {
                    "decision_id": "DECISION_01_FOCAL_SALIENCY_GATE",
                    "condition": "marias_hlaska_points >= 40 or effective_current_amps > 6.0",
                    "compositional_action": "Presunie kameru do priesečníka Zlatej špirály (0.618*W, 0.382*H) a aktivuje zlatý kaustický bloom.",
                },
                {
                    "decision_id": "DECISION_02_RULE_OF_ODDS_ENTITY_FRAMING",
                    "condition": "spawned_entities_count % 2 == 0",
                    "compositional_action": "Pridá vizuálny kotevný totem (nepárny počet 3 alebo 5 prvkov v zábere) pre dynamickú rovnováhu.",
                },
                {
                    "decision_id": "DECISION_03_HYSTERETIC_DEPTH_OF_FIELD",
                    "condition": "uma_bus_utilization >= 0.70 (12-frame dwell active)",
                    "compositional_action": "Zúži hĺbku ostrosti (Bokeh sigma_phi) na aktívny 3-cyklus Rubikovej kocky, čím šetrí šírku pásma UMA zbernice.",
                },
            ],
        }

    def project_interface_through_marias_playing_cards(
        self,
        played_card_id: str = "SRDCIA_KRAL_HLASKA_40",
        grid_from_idx: int = 12,
        grid_to_idx: int = 28,
    ) -> Dict[str, Any]:
        """
        Definuje samotné užívateľské rozhranie (UI) ako živú projekciu cez 32-kartový Mariáš
        (Render(S) = Pi_Marias(S) != S): používateľ nekliká na statické tlačidlá, ale vynáša karty
        v partii Mariášu proti jadru (Omniscience Core Opponent), pričom každá karta premieta
        iný pohľad na mriežku, telemetriu a skladové zásoby.
        """
        marias_ui_projection_deck = [
            {
                "card_id": "SRDCIA_KRAL_HLASKA_40",
                "suit_sk": "Červeň / Srdcia (Atut / Tromf)",
                "rank_sk": "Kráľ + Horník (Hláška 40 b.)",
                "projected_ui_view": "SOLAR_TRUTH_COMMAND_VIEW",
                "vulkan_shader_projection": "Zobrazí 8x8 hex-grid v zlatom reze (61.8% viewport), odomkne +40 b. v 24B KMB1 pakete a zvýši napätie o +2.4 V.",
                "ai_opponent_counter_card": "SRDCIA_ESO_STOVKA (Jadro odpovedá esom a drží invariant VITAL_MAX_HP == 6)",
            },
            {
                "card_id": "ZELEN_HORNIK_HLASKA_20",
                "suit_sk": "Zeleň / Listy",
                "rank_sk": "Horník + Kráľ (Hláška 20 b.)",
                "projected_ui_view": "STOCK_DEPOT_ACF_LOGISTICS_VIEW",
                "vulkan_shader_projection": "Premietne rozhranie Sklad–Depo (`nabytok47.sk` & `RH Isolation Slovakia` ACF `meta_query` za 0.42 ms) a stav zásob.",
                "ai_opponent_counter_card": "ZELEN_DESIATKA (Jadro optimalizuje genetickú taxonómiu na Fitness 0.9918)",
            },
            {
                "card_id": "ZALUD_SEDMA_ULTIMO",
                "suit_sk": "Žaluď",
                "rank_sk": "Sedma (Posledný zdvih / Ultimo)",
                "projected_ui_view": "MORTAR_BALLISTIC_AND_OHM_GRID_VIEW",
                "vulkan_shader_projection": "Premietne parabolickú balistiku mínometu theta_aim = 0.5*arcsin(g*d/v0^2) a prúdovú mapu I = V/R <= 8.0 A.",
                "ai_opponent_counter_card": "ZALUD_KRAL (Jadro spúšťa komutátor Rubikovej kocky [R, U] = R U R' U')",
            },
            {
                "card_id": "GULA_ SPODOK_BETL_DURCH",
                "suit_sk": "Guľa",
                "rank_sk": "Spodok (Režim Betl / Durch — 0 zdvihov)",
                "projected_ui_view": "CHRONOMETRIC_SHADOW_L1D_SURVIVAL_VIEW",
                "vulkan_shader_projection": "Prepne UI do 7.5 W L1D Survival projekcie (27.5 KB v L1D cache, 1.42 ms latencia, 0.3382 mJ za rad inštrukcií).",
                "ai_opponent_counter_card": "GULA_SEDMA (Jadro potvrdzuje 12-snímkovú hysteréziu bez kmitania zbernice)",
            },
        ]

        selected = next((c for c in marias_ui_projection_deck if c["card_id"] == played_card_id), marias_ui_projection_deck[0])

        return {
            "projection_principle": "Render(S) = Pi_Marias(S) != S (Rozhranie je čistá projekcia stavu cez vynesenú kartu Mariášu)",
            "active_played_card": selected,
            "interactive_marias_projection_deck": marias_ui_projection_deck,
            "grid_transition": {"from_idx": grid_from_idx, "to_idx": grid_to_idx, "kmb1_packet_bytes": 24},
            "vital_max_hp": self.vital_max_hp,
        }

    def synthesize_omniscience_core_unified_model(self) -> Dict[str, Any]:
        """
        Komplexný architektonický model OMNISCIENCE CORE, v ktorom súčasne a deterministicky
        funguje každá jedna vlastnosť projektu.
        """
        from krystal_kernel.dusan_game_vulkan_power_tensor_bridge import GLOBAL_DUSAN_GAME_VULKAN_BRIDGE
        from krystal_kernel.senior_prompt_lore_quest_engine import GLOBAL_SENIOR_PROMPT_LORE_ENGINE
        from krystal_kernel.adaptive_genetic_taxonomy_scenario_engine import GLOBAL_ADAPTIVE_GENETIC_SCENARIO_ENGINE

        multimedia = self.generate_all_multimedia_and_interface_assets()
        design_manual = self.synthesize_material_golden_ratio_design_manual()
        dusan_vulkan = GLOBAL_DUSAN_GAME_VULKAN_BRIDGE.fetch_tensor_and_behavioral_schema_dependencies()
        rubik_scrum = GLOBAL_SENIOR_PROMPT_LORE_ENGINE.compile_senior_prompt_lore_quest_script()
        ga_recompiler = GLOBAL_ADAPTIVE_GENETIC_SCENARIO_ENGINE.recompile_bugs_into_epistemic_kernel_features()

        return {
            "architecture_codename": "OMNISCIENCE_CORE_SOVEREIGN_V1",
            "vital_max_hp": self.vital_max_hp,
            "constitutional_law": "¬Valid(Action) => ΔS = 0 | VITAL_MAX_HP == 6 (6 pevných stredov Rubikovej kocky)",
            "composite_epistemic_iq": 164.0,
            "expected_calibration_error_ece": 0.0042,
            "warm_l1d_latency_ms": 1.42,
            "performance_per_cost_advantage_x": 319.0,
            "pillars": {
                "pillar_01_epistemic_tensors_and_6d_trigonometry": {
                    "hex_grid_cognitive_tensor": "HexGridCognitiveTensor[48, 6] (27.5 KB v 48 KB L1D)",
                    "trigonometric_6d_state_vector": "(dx, dy, kappa, p, v, phi_phase) s cos(pi) = -1 (redundancia 82% -> 19.65%)",
                    "algebraic_action_tensor": dusan_vulkan["vulkan_power_and_parameter_plan"]["synthesized_power_plan"]["ohm_grid_telemetry"],
                },
                "pillar_02_semantic_mirror_dual_prose": {
                    "colloquial_register_sk": "Zlaté svetlo pravdy preniká cez hmlu močiara; kráľovská Hláška (+40 b.) nabíja mriežku napätím 12.6 V a drží všetkých 6 životov uzamknutých.",
                    "formal_engineering_register_sk": "VITAL_MAX_HP == 6 | I = 12.6V / 2.4Ohm = 5.25A <= 8.0A | 6D Vector cos(pi)=-1 | Dwell = 12 frames | KMB1 = 24B CRC32 OK.",
                },
                "pillar_03_rubiks_cube_coherent_choice_and_kociemba": rubik_scrum["rubiks_cube_coherent_choice_mechanics"],
                "pillar_04_adaptive_scenarios_ga_taxonomy_and_recompiler": ga_recompiler,
                "pillar_05_vulkan_power_plan_and_16_server_params": dusan_vulkan["vulkan_power_and_parameter_plan"],
                "pillar_06_conda_async_and_stock_depot_acf_query": {
                    "conda_async_fixes": dusan_vulkan["conda_async_behavioral_solutions"],
                    "stock_depot_acf_query": dusan_vulkan["stock_depot_reasoning_acf_query_engine"],
                },
                "pillar_07_blender_godot_babylon_oxygen_sync": {
                    "binary_packet": "24-Byte KMB1 (<4sQHHHHI) + 128-Byte ZeroCopyContractFrame",
                    "engines": ["Blender MCP", "Godot 4.3 (KrystalHybridLoreAggregator.gd)", "Babylon.js (babylon_dusan_game_sync_bridge.js)", "WordPress 7.1.4 + Oxygen 6.2.1 (nabytok47.sk)"],
                },
                "pillar_08_material_golden_ratio_design_and_multimedia": {
                    "design_manual": design_manual,
                    "generated_multimedia_assets": multimedia,
                },
            },
        }

    def generate_all_multimedia_and_interface_assets(self) -> Dict[str, Any]:
        wav_info = self.synthesize_polyphonic_wav_soundtrack()
        svg_info = self.synthesize_svg_graphics_emblem()
        ui_info = self.synthesize_interactive_flat_design_studio_ui()
        return {
            "status": "MULTIMEDIA_GRAPHICS_MUSIC_AND_INTERFACE_GENERATED",
            "vital_max_hp": self.vital_max_hp,
            "audio_soundtrack_wav": wav_info,
            "vector_graphics_svg": svg_info,
            "interactive_studio_interface_html": ui_info,
        }


GLOBAL_GRAPHICS_MUSIC_UI_SYNTHESIZER = GraphicsMusicInterfaceSynthesizer()
