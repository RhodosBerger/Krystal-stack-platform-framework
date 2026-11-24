/*
 * krystal_state_before_after_scaler.c
 *
 * Algoritmy v C terminológii (`state_a`, `state_before`, `state_after`), ktoré:
 *   1. Zachovávajú ústavný invariant `VITAL_MAX_HP == 6` (`!valid => state_after = state_before`),
 *   2. Škálujú rastrovú projekciu z `state_before` na cieľový formát v `state_after`,
 *   3. Vkladajú v opakovacích slučkách (`raster_loop_epoch`) procedurálne objekty z knižnice
 *      katalógu (odpojeného od stereotypu Anno 2070 do 60s Cyberpunk × Krystal Adrenalin estetiky).
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define KRYSTAL_VITAL_MAX_HP 6u
#define KRYSTAL_MAX_W 64u
#define KRYSTAL_MAX_H 36u

typedef struct {
    uint32_t prop_guid;
    uint8_t  pos_x_pct;
    uint8_t  pos_y_pct;
    uint8_t  radius_px;
    uint8_t  palette_rgb[3];
    uint16_t harmonic_lobes;
} KrystalProceduralProp;

typedef struct {
    uint32_t vital_hp;
    uint32_t uma_milli;          /* < 780 pre otvorený tieňový výpočet */
    uint32_t raster_loop_epoch;
    uint32_t active_props_count;
} KrystalRenderStateA;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t state_crc;
    uint8_t  rgb[KRYSTAL_MAX_H][KRYSTAL_MAX_W][3];
} KrystalRenderStateBefore;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t state_crc;
    uint32_t applied_props;
    uint8_t  committed;
    uint8_t  rgb[KRYSTAL_MAX_H][KRYSTAL_MAX_W][3];
} KrystalRenderStateAfter;

static uint32_t krystal_fnv1a_frame(const uint8_t *buf, size_t len) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < len; ++i) {
        h ^= (uint32_t)buf[i];
        h *= 16777619u;
    }
    return h;
}

int krystal_scale_and_inject_procedural_props(
    const KrystalRenderStateA *state_a,
    const KrystalRenderStateBefore *state_before,
    KrystalRenderStateAfter *state_after,
    uint32_t target_w,
    uint32_t target_h,
    const KrystalProceduralProp *props,
    uint32_t props_len
) {
    if (!state_a || !state_before || !state_after) {
        return -1;
    }
    /* Ústavná kontrola: ak akcia poruší VITAL_MAX_HP == 6 alebo prekročí hranice, S_{t+1} == S_t */
    if (state_a->vital_hp != KRYSTAL_VITAL_MAX_HP ||
        target_w == 0 || target_w > KRYSTAL_MAX_W ||
        target_h == 0 || target_h > KRYSTAL_MAX_H ||
        state_before->width == 0 || state_before->height == 0) {
        state_after->width = state_before->width;
        state_after->height = state_before->height;
        state_after->state_crc = state_before->state_crc;
        state_after->applied_props = 0u;
        state_after->committed = 0u;
        memcpy(state_after->rgb, state_before->rgb, sizeof(state_before->rgb));
        return -2;
    }

    memset(state_after->rgb, 0, sizeof(state_after->rgb));
    state_after->width = target_w;
    state_after->height = target_h;

    /* C-štýl pevno-rádového (16.16 fixed-point) rastrového škálovania so slučkou opakovania */
    const uint32_t step_x_fp = (state_before->width << 16) / target_w;
    const uint32_t step_y_fp = (state_before->height << 16) / target_h;
    const uint32_t phase_shift = state_a->raster_loop_epoch & 7u;

    for (uint32_t y = 0; y < target_h; ++y) {
        uint32_t sy = ((y * step_y_fp) >> 16);
        if (sy >= state_before->height) sy = state_before->height - 1u;
        for (uint32_t x = 0; x < target_w; ++x) {
            uint32_t sx = (((x * step_x_fp) >> 16) + phase_shift) % state_before->width;
            state_after->rgb[y][x][0] = state_before->rgb[sy][sx][0];
            state_after->rgb[y][x][1] = state_before->rgb[sy][sx][1];
            state_after->rgb[y][x][2] = state_before->rgb[sy][sx][2];
        }
    }

    /* Vloženie procedurálnych objektov z knižnice (60s Cyberpunk x Krystal Adrenalin) */
    uint32_t applied = 0u;
    for (uint32_t p = 0; p < props_len; ++p) {
        uint32_t cx = (props[p].pos_x_pct * (target_w - 1u)) / 100u;
        uint32_t cy = (props[p].pos_y_pct * (target_h - 1u)) / 100u;
        int32_t rad = (int32_t)props[p].radius_px;
        for (int32_t dy = -rad; dy <= rad; ++dy) {
            for (int32_t dx = -rad; dx <= rad; ++dx) {
                if (dx * dx + dy * dy <= rad * rad) {
                    int32_t px = (int32_t)cx + dx;
                    int32_t py = (int32_t)cy + dy;
                    if (px >= 0 && px < (int32_t)target_w && py >= 0 && py < (int32_t)target_h) {
                        state_after->rgb[py][px][0] = props[p].palette_rgb[0];
                        state_after->rgb[py][px][1] = props[p].palette_rgb[1];
                        state_after->rgb[py][px][2] = props[p].palette_rgb[2];
                    }
                }
            }
        }
        ++applied;
    }

    state_after->applied_props = applied;
    state_after->committed = 1u;
    state_after->state_crc = krystal_fnv1a_frame(
        &state_after->rgb[0][0][0],
        sizeof(state_after->rgb)
    );
    return 0;
}

int main(void) {
    KrystalRenderStateA state_a = { KRYSTAL_VITAL_MAX_HP, 420u, 3u, 3u };
    KrystalRenderStateBefore state_before;
    KrystalRenderStateAfter state_after;
    memset(&state_before, 0, sizeof(state_before));
    state_before.width = 16u;
    state_before.height = 12u;

    for (uint32_t y = 0; y < state_before.height; ++y) {
        for (uint32_t x = 0; x < state_before.width; ++x) {
            state_before.rgb[y][x][0] = (uint8_t)(10u + ((x + y) & 1u) * 18u);
            state_before.rgb[y][x][1] = (uint8_t)(14u + ((x + y) & 1u) * 64u);
            state_before.rgb[y][x][2] = (uint8_t)(23u + ((x + y) & 1u) * 96u);
        }
    }
    state_before.state_crc = krystal_fnv1a_frame(&state_before.rgb[0][0][0], sizeof(state_before.rgb));

    const KrystalProceduralProp catalog_props[3] = {
        { 0x60A101u, 50u, 30u, 2u, { 0u, 229u, 255u }, 6u },   /* Crystal Obelisk */
        { 0x60A102u, 70u, 48u, 2u, { 52u, 211u, 153u }, 5u },  /* Druid Relay Totem */
        { 0x60A103u, 44u, 60u, 2u, { 255u, 158u, 44u }, 4u }   /* 60s Apollo Extractor */
    };

    int rc = krystal_scale_and_inject_procedural_props(
        &state_a, &state_before, &state_after, 48u, 28u, catalog_props, 3u
    );
    printf(
        "{\"rc\":%d,\"vital_hp\":%u,\"before_wh\":[%u,%u],\"after_wh\":[%u,%u],"
        "\"applied_props\":%u,\"committed\":%u,\"after_crc_hex\":\"0x%08x\"}\n",
        rc, state_a.vital_hp, state_before.width, state_before.height,
        state_after.width, state_after.height, state_after.applied_props,
        state_after.committed, state_after.state_crc
    );
    return (rc == 0 && state_after.committed == 1u) ? 0 : 1;
}
