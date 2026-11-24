// GNOME Shell 46+ ESM rozšírenie pre Krystal Stack (GDM + Wayland + WSLg + Godot 4 WebGL)
// Ústavný invariant: VITAL_MAX_HP == 6 | UMA < 0.78
import GLib from 'gi://GLib';

export const WINDOW_MODES = [
    'MODE_01_COSMIC_AUTO_TILING',
    'MODE_02_NIRI_SCROLLABLE_STRIP',
    'MODE_03_HYPRLAND_DWINDLE_SHADER',
    'MODE_04_STACKED_TABBED_DECK',
    'MODE_05_FLOATING_HUD_OVERLAY',
];

export default class KrystalSovereignGnomeExtension {
    enable() {
        this._modeIndex = 0;
        this._vitalMaxHp = 6;
    }

    cycleWindowMode() {
        this._modeIndex = (this._modeIndex + 1) % WINDOW_MODES.length;
        return WINDOW_MODES[this._modeIndex];
    }

    launchGodotWslProcedure(projectDir) {
        const cmd = `godot4 --rendering-driver opengl3 --path "${projectDir}"`;
        GLib.spawn_command_line_async(cmd);
        return { launched: true, vital_max_hp: this._vitalMaxHp, command: cmd };
    }

    disable() {
        this._modeIndex = 0;
    }
}
