/**
 * KRYSTAL-STACK FRONTEND <-> BACKEND HYBRID OBJECTIVE & MARIÁŠ FETCH CLIENT
 * Connects Web Hub UI to Local Validated Micro-Server / Master Hub endpoints:
 *   - POST /api/hybrid/fetch_objectives (Strapi CMS + Medusa/Woo + V-Ray Signal + 8x8 Grid Binary)
 *   - POST /api/hybrid/marias_click (Mariáš marriage/turn staging + Semantic Click Animation)
 */

export class KrystalHybridFrontendClient {
  constructor(baseUrl = "http://127.0.0.1:8080") {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.vitalMaxHp = 6;
  }

  async fetchObjectives({
    questSlug = "citadel-well-of-souls",
    choiceId = "init_harmonic",
    newGameSeed = "",
    gridMove = "0,0:2,3",
    renderTerms = {
      direct_irradiance: 1.35,
      gi_bounce_luminance: 0.62,
      specular_roughness: 0.24,
      caustic_photon_flux: 0.48,
      mortar_elevation_deg: 42.5,
    },
  } = {}) {
    const res = await fetch(`${this.baseUrl}/api/hybrid/fetch_objectives`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        quest_slug: questSlug,
        choice_id: choiceId,
        new_game_seed: newGameSeed,
        grid_move: gridMove,
        render_terms: renderTerms,
      }),
    });
    if (!res.ok) throw new Error(`Hybrid fetch failed: ${res.status}`);
    return await res.json();
  }

  async triggerMariasCardClick({
    suit = "HEARTS_SRDCIA",
    rank = "KING_KRAL",
    questSlug = "citadel-well-of-souls",
  } = {}) {
    const res = await fetch(`${this.baseUrl}/api/hybrid/marias_click`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ suit, rank, quest_slug: questSlug }),
    });
    if (!res.ok) throw new Error(`Mariáš click failed: ${res.status}`);
    return await res.json();
  }
}
