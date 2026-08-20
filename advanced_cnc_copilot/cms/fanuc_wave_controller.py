"""
FANUC Wave Controller
KrystalStack Architecture Implementation

PARADIGM: Wave Computing for CNC Control
- Machine vibration treated as Visual Entropy
- G-Code toolpaths as Wave Equations
- Process scheduling via Gravitational Pull
- Chatter suppression through Entropy Thresholds

ARCHITECTURE LAYER: Bridge between GAMESA/KrystalStack and FANUC Physical Hardware
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import math

# KrystalStack Imports (First Layer)
from cms.active_optic_compositor import ActiveOpticCompositor, EntropyMetrics
from cms.data_synchronizer import DataSynchronizer, AngleOfView
from cms.process_gravitator import ProcessGravitator, GravitationalProcess


@dataclass
class WaveFunction:
    """
    Represents a toolpath segment as a Wave Equation
    
    NOT a line segment - a Wave with:
    - Amplitude (cutting depth)
    - Frequency (feed rate oscillation)
    - Phase (position in cycle)
    - Coherence (quality of cut)
    """
    amplitude: float  # Cutting depth/force
    frequency: float  # Feed rate oscillation
    phase: float  # Current position in wave cycle (0-2π)
    wavelength: float  # Distance per cycle
    coherence: float  # Wave stability (0.0-1.0)
    
    def interference_with(self, other: 'WaveFunction') -> float:
        """
        Calculate interference pattern between two waves
        
        Constructive interference (aligned) = stable cutting
        Destructive interference (opposed) = chatter
        
        Returns:
            Interference coefficient: -1.0 (destructive) to 1.0 (constructive)
        """
        phase_diff = abs(self.phase - other.phase)
        freq_ratio = self.frequency / max(other.frequency, 0.001)
        
        # Constructive when phases align and frequencies match
        phase_alignment = math.cos(phase_diff)
        freq_harmony = 1.0 / (1.0 + abs(1.0 - freq_ratio))
        
        return phase_alignment * freq_harmony
    
    def propagate(self, time_delta: float) -> 'WaveFunction':
        """
        Propagate wave forward in time
        
        Args:
            time_delta: Time step (seconds)
        
        Returns:
            New wave state after propagation
        """
        # Phase advances based on frequency
        new_phase = (self.phase + 2 * math.pi * self.frequency * time_delta) % (2 * math.pi)
        
        # Coherence degrades with time (entropy increases)
        coherence_decay = 0.99  # 1% decay per step
        new_coherence = self.coherence * coherence_decay
        
        return WaveFunction(
            amplitude=self.amplitude,
            frequency=self.frequency,
            phase=new_phase,
            wavelength=self.wavelength,
            coherence=new_coherence
        )


class ToolpathHologram:
    """
    Holographic representation of toolpath
    
    NOT a sequence of XYZ coordinates - a 3D interference pattern
    Each point in space has a "Potential Field" from all wave sources
    
    METAPHOR: Like a hologram, cutting information is distributed
    across the entire pattern, not localized to single points
    """
    
    def __init__(self, resolution: Tuple[int, int, int] = (64, 64, 64)):
        """
        Initialize holographic space
        
        Args:
            resolution: Hex grid resolution (x, y, z)
        """
        self.resolution = resolution
        
        # Hex grid for holographic data
        # Each cell stores wave interference amplitude
        self.hologram = np.zeros(resolution, dtype=np.complex128)
        
        # Wave sources (tool positions emitting waves)
        self.wave_sources: List[Tuple[np.ndarray, WaveFunction]] = []
    
    def add_wave_source(self, position: np.ndarray, wave: WaveFunction):
        """
        Add wave source to hologram
        
        Args:
            position: 3D position (x, y, z)
            wave: Wave function at this position
        """
        self.wave_sources.append((position, wave))
        
        # Regenerate hologram with new source
        self._regenerate_hologram()
    
    def _regenerate_hologram(self):
        """
        Regenerate holographic interference pattern
        
        Each point in space receives waves from all sources
        Waves interfere constructively or destructively
        """
        # Create coordinate grid
        x = np.linspace(0, 1, self.resolution[0])
        y = np.linspace(0, 1, self.resolution[1])
        z = np.linspace(0, 1, self.resolution[2])
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.stack([X, Y, Z], axis=-1)
        
        # Reset hologram
        self.hologram = np.zeros(self.resolution, dtype=np.complex128)
        
        # Add interference from each wave source
        for source_pos, wave in self.wave_sources:
            # Distance from source to each grid point
            distances = np.linalg.norm(grid_points - source_pos, axis=-1)
            
            # Wave amplitude decays with distance
            amplitudes = wave.amplitude / (1.0 + distances / wave.wavelength)
            
            # Phase shift based on distance
            phases = wave.phase + 2 * math.pi * distances / wave.wavelength
            
            # Complex wave representation
            wave_contribution = amplitudes * np.exp(1j * phases)
            
            # Add to hologram (interference)
            self.hologram += wave_contribution
        
    def sample_at(self, position: np.ndarray) -> complex:
        """
        Sample hologram at specific position
        
        Returns:
            Complex amplitude at position
        """
        # Convert position to grid indices
        idx = (position * np.array(self.resolution)).astype(int)
        idx = np.clip(idx, 0, np.array(self.resolution) - 1)
        
        return self.hologram[idx[0], idx[1], idx[2]]
    
    def get_entropy_map(self) -> np.ndarray:
        """
        Calculate spatial entropy distribution
        
        High entropy regions = unstable cutting zones
        Low entropy regions = stable cutting zones
        
        Returns:
            3D entropy map (0.0-1.0)
        """
        # Entropy based on amplitude variance in local neighborhoods
        # High variance = high entropy = chatter risk
        
        amplitude = np.abs(self.hologram)
        
        # Local variance (3x3x3 kernel)
        from scipy.ndimage import generic_filter
        
        def local_variance(window):
            return np.var(window)
        
        entropy = generic_filter(amplitude, local_variance, size=3)
        
        # Normalize to 0-1
        entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-10)
        
        return entropy


class FanucFOCAS_Source:
    """
    Data source for FANUC FOCAS data
    
    Feeds into DataSynchronizer as an "Angle of View"
    Provides: Spindle Load, Servo Error, Axis Position
    """
    
    def __init__(self, focas_handle):
        """
        Initialize FOCAS data source
        
        Args:
            focas_handle: FOCAS library handle (pyfocas)
        """
        self.handle = focas_handle
        self.last_read_time = datetime.now()
    
    def create_angle_of_view(self) -> AngleOfView:
        """
        Create AngleOfView for DataSynchronizer
        
        Returns:
            AngleOfView with current FOCAS data
        """
        # Read FOCAS data
        spindle_load = self._read_spindle_load()
        servo_error = self._read_servo_error()
        axis_position = self._read_axis_position()
        
        # Package as AngleOfView
        return AngleOfView(
            source_id='fanuc_focas',
            timestamp=datetime.now(),
            data={
                'spindle_load': spindle_load,
                'servo_error': servo_error,
                'axis_position': axis_position,
            },
            confidence=1.0,  # FOCAS is authoritative source
            latency_ms=(datetime.now() - self.last_read_time).total_seconds() * 1000
        )
    
    def _read_spindle_load(self) -> float:
        """Read spindle load percentage (0-100)"""
        try:
            import pyfocas
            # FOCAS API call
            load = pyfocas.cnc_rdspload(self.handle, 0)  # 0 = no spindle
            return float(load.data.spload.data)
        except:
            return 0.0
    
    def _read_servo_error(self) -> Dict[str, float]:
        """Read servo error for all axes"""
        try:
            import pyfocas
            errors = {}
            for axis in ['X', 'Y', 'Z']:
                err = pyfocas.cnc_rdsverr(self.handle, axis)
                errors[axis] = float(err)
            return errors
        except:
            return {'X': 0.0, 'Y': 0.0, 'Z': 0.0}
    
    def _read_axis_position(self) -> Dict[str, float]:
        """Read current axis positions"""
        try:
            import pyfocas
            pos = pyfocas.cnc_rdposition(self.handle, -1)  # -1 = all axes
            return {
                'X': float(pos.data.abs.data[0]) / 1000.0,  # Convert to mm
                'Y': float(pos.data.abs.data[1]) / 1000.0,
                'Z': float(pos.data.abs.data[2]) / 1000.0,
            }
        except:
            return {'X': 0.0, 'Y': 0.0, 'Z': 0.0}


class ChatterEntropyDetector:
    """
    Detects chatter by treating it as Visual Entropy
    
    Machine vibration = Visual chaos
    High servo error variance = High entropy
    High entropy = Throttle feed rate
    
    Uses ActiveOpticCompositor from KrystalStack
    """
    
    def __init__(self, entropy_threshold: float = 0.7):
        """
        Initialize chatter detector
        
        Args:
            entropy_threshold: Entropy above this triggers suppression
        """
        self.compositor = ActiveOpticCompositor()
        self.entropy_threshold = entropy_threshold
        
        # History for entropy calculation
        self.servo_error_history: List[Dict[str, float]] = []
        self.max_history = 100
    
    def detect_chatter(self, servo_error: Dict[str, float]) -> Tuple[bool, EntropyMetrics]:
        """
        Detect chatter from servo error data
        
        Args:
            servo_error: Current servo errors {axis: error}
        
        Returns:
            (is_chattering, entropy_metrics)
        """
        # Add to history
        self.servo_error_history.append(servo_error)
        if len(self.servo_error_history) > self.max_history:
            self.servo_error_history.pop(0)
        
        # Convert servo error to "visual" representation
        # Treat each axis as a color channel (R, G, B)
        visual_data = self._servo_to_visual(self.servo_error_history)
        
        # Calculate entropy using ActiveOpticCompositor
        entropy_metrics = self.compositor.calculate_entropy(visual_data)
        
        # Chatter detected if entropy exceeds threshold
        is_chattering = entropy_metrics.total_entropy > self.entropy_threshold
        
        return is_chattering, entropy_metrics
    
    def _servo_to_visual(self, servo_history: List[Dict[str, float]]) -> np.ndarray:
        """
        Convert servo error history to visual representation
        
        Each axis becomes a color channel:
        X → Red
        Y → Green  
        Z → Blue
        
        Returns:
            Visual array (height=time, width=1, channels=3)
        """
        height = len(servo_history)
        visual = np.zeros((height, 1, 3), dtype=np.float32)
        
        for i, errors in enumerate(servo_history):
            visual[i, 0, 0] = errors.get('X', 0.0)  # Red = X
            visual[i, 0, 1] = errors.get('Y', 0.0)  # Green = Y
            visual[i, 0, 2] = errors.get('Z', 0.0)  # Blue = Z
        
        # Normalize to 0-1
        visual = np.abs(visual)
        visual = visual / (np.max(visual) + 1e-10)
        
        return visual


class FanucWaveController:
    """
    Main Wave Controller for FANUC CNC
    
    ARCHITECTURE:
    - Data flows through DataSynchronizer (multiple angles of view)
    - Vibration analyzed as Visual Entropy (ActiveOpticCompositor)
    - Processes scheduled via Gravitational Pull (ProcessGravitator)
    - Toolpath represented as Wave Hologram
    
    REAL-TIME BEHAVIOR:
    - E-Stop thread: Highest gravitational mass (always wins CPU)
    - Chatter detection: Medium gravity
    - Data logging: Low gravity
    """
    
    def __init__(self, focas_handle):
        """
        Initialize FANUC Wave Controller
        
        Args:
            focas_handle: FOCAS connection handle
        """
        # Data sources
        self.focas_source = FanucFOCAS_Source(focas_handle)
        self.synchronizer = DataSynchronizer()
        
        # Entropy-based chatter detection
        self.chatter_detector = ChatterEntropyDetector(entropy_threshold=0.7)
        
        # Toolpath hologram
        self.hologram = ToolpathHologram(resolution=(64, 64, 64))
        
        # Process gravitator for scheduling
        self.gravitator = ProcessGravitator()
        
        # Current wave state
        self.current_wave = WaveFunction(
            amplitude=1.0,
            frequency=1.0,
            phase=0.0,
            wavelength=1.0,
            coherence=1.0
        )
        
        # Feed rate override (1.0 = 100%)
        self.feed_override = 1.0
        
        # Setup gravitational processes
        self._setup_processes()
    
    def _setup_processes(self):
        """
        Setup processes with gravitational scheduling
        
        Mass (Priority):
        - E-Stop Monitor: 1000.0 (Heavy = High priority)
        - Chatter Suppression: 100.0 (Medium)
        - Data Logging: 1.0 (Light = Low priority)
        """
        # E-Stop monitor (highest priority)
        self.gravitator.add_process(GravitationalProcess(
            process_id='e_stop_monitor',
            mass=1000.0,  # Massive = highest priority
            velocity=1.0,
            core_affinity=0,  # Pin to core 0
            callback=self._e_stop_monitor
        ))
        
        # Chatter suppression (medium priority)
        self.gravitator.add_process(GravitationalProcess(
            process_id='chatter_suppression',
            mass=100.0,
            velocity=1.0,
            core_affinity=1,
            callback=self._chatter_suppression
        ))
        
        # Data logging (low priority)
        self.gravitator.add_process(GravitationalProcess(
            process_id='data_logging',
            mass=1.0,  # Light = lowest priority
            velocity=1.0,
            core_affinity=2,
            callback=self._data_logging
        ))
    
    def update_toolpath_wave(self, gcode_line: str):
        """
        Convert G-Code to Wave Equation
        
        Instead of parsing as coordinates, interpret as wave parameters
        
        Args:
            gcode_line: G-Code command (e.g., "G01 X100 Y50 F500")
        """
        # Parse G-Code (simplified)
        parts = gcode_line.split()
        
        # Extract parameters
        x = y = z = f = None
        for part in parts:
            if part.startswith('X'):
                x = float(part[1:])
            elif part.startswith('Y'):
                y = float(part[1:])
            elif part.startswith('Z'):
                z = float(part[1:])
            elif part.startswith('F'):
                f = float(part[1:])
        
        if x is None or y is None or z is None:
            return
        
        # Convert to wave parameters
        # Amplitude = change in Z (cutting depth)
        amplitude = abs(z) if z else 1.0
        
        # Frequency = feed rate / distance
        distance = math.sqrt(x**2 + y**2 + z**2)
        frequency = (f / 60.0) / distance if f and distance > 0 else 1.0  # Hz
        
        # Create wave
        wave = WaveFunction(
            amplitude=amplitude,
            frequency=frequency,
            phase=0.0,
            wavelength=distance,
            coherence=1.0
        )
        
        # Add to hologram
        position = np.array([x / 1000.0, y / 1000.0, z / 1000.0])  # Normalize
        self.hologram.add_wave_source(position, wave)
        
        # Update current wave
        self.current_wave = wave
    
    def step(self, time_delta: float = 0.001):
        """
        Single timestep of wave controller
        
        Args:
            time_delta: Time step (seconds)
        """
        # Get synchronized data from all sources
        focas_view = self.focas_source.create_angle_of_view()
        self.synchronizer.add_view(focas_view)
        
        consolidated = self.synchronizer.consolidate()
        
        # Extract data
        spindle_load = consolidated.get('spindle_load', 0.0)
        servo_error = consolidated.get('servo_error', {})
        
        # Detect chatter using entropy
        is_chattering, entropy_metrics = self.chatter_detector.detect_chatter(servo_error)
        
        # If chattering, suppress via feed override
        if is_chattering:
            # Reduce feed rate proportional to entropy
            suppression_factor = 1.0 - (entropy_metrics.total_entropy - 0.7) / 0.3
            suppression_factor = max(0.5, min(1.0, suppression_factor))  # 50-100%
            
            self.feed_override = suppression_factor
        else:
            # Gradually restore feed rate
            self.feed_override = min(1.0, self.feed_override + 0.01)
        
        # Propagate wave
        self.current_wave = self.current_wave.propagate(time_delta)
        
        # Schedule processes via gravitation
        self.gravitator.schedule_next()
    
    def _e_stop_monitor(self):
        """E-Stop monitor process (highest priority)"""
        # Check for emergency conditions
        # Would trigger E-Stop via FOCAS
        pass
    
    def _chatter_suppression(self):
        """Chatter suppression process (medium priority)"""
        # Adjust feed override
        # Would send override command via FOCAS
        pass
    
    def _data_logging(self):
        """Data logging process (lowest priority)"""
        # Log current state
        pass
    
    def get_entropy_map(self) -> np.ndarray:
        """
        Get spatial entropy map of toolpath
        
        Shows which regions of cut are unstable
        
        Returns:
            3D entropy map
        """
        return self.hologram.get_entropy_map()
    
    def get_feed_override(self) -> float:
        """
        Get current feed rate override
        
        Returns:
            Override factor (0.5-1.0)
        """
        return self.feed_override
