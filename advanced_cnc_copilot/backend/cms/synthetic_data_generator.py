"""
Synthetic Manufacturing Data Generator
Generates realistic CNC operation data for replacement parts production

Strategies:
1. Physics-based simulation
2. Industry standard templates
3. Parametric variation
4. Failure mode injection
5. Public dataset mining
6. LLM-based generation
"""

import random
import numpy as np
import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json


class MaterialDatabase:
    """Database of material properties for realistic simulation"""
    
    MATERIALS = {
        '6061-T6 Aluminum': {
            'specific_cutting_force': 700,  # N/mm²
            'density': 2.70,  # g/cm³
            'hardness_hb': 95,
            'cost_per_kg': 4.50,
            'cutting_speed_m_min': 300,
            'feed_per_tooth': 0.15,
            'chip_thinning_factor': 0.8
        },
        '7075-T6 Aluminum': {
            'specific_cutting_force': 950,
            'density': 2.81,
            'hardness_hb': 150,
            'cost_per_kg': 8.50,
            'cutting_speed_m_min': 200,
            'feed_per_tooth': 0.12,
            'chip_thinning_factor': 0.75
        },
        '304 Stainless Steel': {
            'specific_cutting_force': 2100,
            'density': 8.00,
            'hardness_hb': 180,
            'cost_per_kg': 6.00,
            'cutting_speed_m_min': 80,
            'feed_per_tooth': 0.08,
            'chip_thinning_factor': 0.85
        },
        '1018 Steel': {
            'specific_cutting_force': 1800,
            'density': 7.87,
            'hardness_hb': 120,
            'cost_per_kg': 1.20,
            'cutting_speed_m_min': 150,
            'feed_per_tooth': 0.10,
            'chip_thinning_factor': 0.80
        },
        '4140 Steel': {
            'specific_cutting_force': 2200,
            'density': 7.85,
            'hardness_hb': 250,
            'cost_per_kg': 2.50,
            'cutting_speed_m_min': 100,
            'feed_per_tooth': 0.08,
            'chip_thinning_factor': 0.90
        },
        'Ti-6Al-4V Titanium': {
            'specific_cutting_force': 1300,
            'density': 4.43,
            'hardness_hb': 340,
            'cost_per_kg': 35.00,
            'cutting_speed_m_min': 60,
            'feed_per_tooth': 0.05,
            'chip_thinning_factor': 0.70
        }
    }


class PartTemplateGenerator:
    """
    Generates replacement part specifications from templates
    """
    
    PART_TEMPLATES = {
        'bearing_housing': {
            'base_dimensions': {'outer_diameter': (50, 200), 'bore_diameter': (20, 100), 'width': (30, 80)},
            'typical_material': '6061-T6 Aluminum',
            'features': ['face', 'bore', 'bolt_circle', 'chamfer'],
            'tolerance_class': 'H7',
            'surface_finish': 'Ra 1.6'
        },
        'shaft': {
            'base_dimensions': {'diameter': (10, 100), 'length': (50, 500)},
            'typical_material': '4140 Steel',
            'features': ['turning', 'threading', 'keyway', 'shoulder'],
            'tolerance_class': 'h6',
            'surface_finish': 'Ra 0.8'
        },
        'bracket': {
            'base_dimensions': {'length': (50, 300), 'width': (30, 200), 'thickness': (5, 20)},
            'typical_material': '6061-T6 Aluminum',
            'features': ['holes', 'pockets', 'edge_machining'],
            'tolerance_class': '±0.1mm',
            'surface_finish': 'Ra 3.2'
        },
        'pulley': {
            'base_dimensions': {'outer_diameter': (100, 400), 'bore_diameter': (20, 80), 'width': (20, 100)},
            'typical_material': '7075-T6 Aluminum',
            'features': ['groove', 'bore', 'face'],
            'tolerance_class': 'H8',
            'surface_finish': 'Ra 0.4'
        },
        'gear': {
            'base_dimensions': {'pitch_diameter': (50, 300), 'face_width': (10, 50), 'bore': (15, 60)},
            'typical_material': '4140 Steel',
            'features': ['teeth', 'bore', 'face', 'chamfer'],
            'tolerance_class': 'DIN 5',
            'surface_finish': 'Ra 0.8'
        }
    }
    
    def generate_part(self, part_type: str, vendor_requirements: Optional[Dict] = None) -> Dict:
        """
        Generate part specification from template
        
        Args:
            part_type: Type of part to generate
            vendor_requirements: Optional vendor-specific requirements
        
        Returns:
            Complete part specification
        """
        if part_type not in self.PART_TEMPLATES:
            raise ValueError(f"Unknown part type: {part_type}")
        
        template = self.PART_TEMPLATES[part_type]
        
        # Generate dimensions within range
        dimensions = {}
        for dim_name, (min_val, max_val) in template['base_dimensions'].items():
            dimensions[dim_name] = round(random.uniform(min_val, max_val), 2)
        
        # Select material
        material = vendor_requirements.get('material', template['typical_material']) if vendor_requirements else template['typical_material']
        
        # Calculate volume and weight
        volume_cm3 = self._calculate_volume(part_type, dimensions)
        density = MaterialDatabase.MATERIALS[material]['density']
        weight_kg = (volume_cm3 * density) / 1000
        
        # Generate part number
        part_number = self._generate_part_number(part_type)
        
        # Estimate machining operations
        operations = self._estimate_operations(part_type, template['features'], material)
        
        # Calculate total cycle time and cost
        total_cycle_time = sum(op['cycle_time_minutes'] for op in operations)
        material_cost = weight_kg * MaterialDatabase.MATERIALS[material]['cost_per_kg']
        machining_cost = total_cycle_time * 1.5  # $1.50/minute machine time
        total_cost = material_cost + machining_cost + (machining_cost * 0.3)  # 30% overhead
        
        return {
            'part_number': part_number,
            'name': f"{part_type.replace('_', ' ').title()} - Replacement",
            'description': f"Replacement {part_type} for end-of-life original component",
            'part_type': part_type,
            'material': material,
            'dimensions': dimensions,
            'weight_kg': round(weight_kg, 4),
            'tolerances': template['tolerance_class'],
            'surface_finish': template['surface_finish'],
            'operations': operations,
            'estimated_cycle_time_minutes': round(total_cycle_time, 2),
            'estimated_cost_usd': round(total_cost, 2),
            'complexity_score': self._rate_complexity(part_type, len(operations)),
            'end_of_life_reason': random.choice(['wear', 'obsolete', 'failed', 'improved_design'])
        }
    
    def _calculate_volume(self, part_type: str, dimensions: Dict) -> float:
        """Calculate approximate volume in cm³"""
        if part_type == 'bearing_housing':
            # Hollow cylinder
            outer_r = dimensions['outer_diameter'] / 20  # mm to cm
            inner_r = dimensions['bore_diameter'] / 20
            width = dimensions['width'] / 10
            return math.pi * width * (outer_r**2 - inner_r**2)
        
        elif part_type == 'shaft':
            # Solid cylinder
            radius = dimensions['diameter'] / 20
            length = dimensions['length'] / 10
            return math.pi * radius**2 * length
        
        elif part_type == 'bracket':
            # Rectangular with approximate pocket removal
            volume = (dimensions['length'] / 10) * (dimensions['width'] / 10) * (dimensions['thickness'] / 10)
            return volume * 0.7  # 30% removed for pockets
        
        elif part_type == 'pulley':
            # Similar to bearing housing
            outer_r = dimensions['outer_diameter'] / 20
            inner_r = dimensions['bore_diameter'] / 20
            width = dimensions['width'] / 10
            return math.pi * width * (outer_r**2 - inner_r**2) * 0.9  # 10% for groove
        
        elif part_type == 'gear':
            # Approximate as solid cylinder (teeth add ~5%)
            radius = dimensions['pitch_diameter'] / 20
            width = dimensions['face_width'] / 10
            return math.pi * radius**2 * width * 1.05
        
        return 100  # Default
    
    def _generate_part_number(self, part_type: str) -> str:
        """Generate unique part number"""
        prefix = part_type[:3].upper()
        number = random.randint(10000, 99999)
        return f"RP-{prefix}-{number}"
    
    def _estimate_operations(self, part_type: str, features: List[str], material: str) -> List[Dict]:
        """Estimate machining operations required"""
        operations = []
        sequence = 1
        
        material_props = MaterialDatabase.MATERIALS[material]
        
        for feature in features:
            op = self._generate_operation(feature, material, sequence)
            operations.append(op)
            sequence += 1
        
        return operations
    
    def _generate_operation(self, feature: str, material: str, sequence: int) -> Dict:
        """Generate single machining operation"""
        material_props = MaterialDatabase.MATERIALS[material]
        
        # Operation parameters based on feature type
        operations_db = {
            'face': {
                'type': 'milling',
                'tool': 'Face Mill Ø100mm',
                'tool_diameter_mm': 100,
                'doc_mm': 2.0,
                'base_time': 5
            },
            'bore': {
                'type': 'boring',
                'tool': 'Boring Bar',
                'tool_diameter_mm': 25,
                'doc_mm': 0.5,
                'base_time': 8
            },
            'holes': {
                'type': 'drilling',
                'tool': 'Drill Ø10mm',
                'tool_diameter_mm': 10,
                'doc_mm': 30,
                'base_time': 3
            },
            'pockets': {
                'type': 'milling',
                'tool': 'End Mill Ø12mm',
                'tool_diameter_mm': 12,
                'doc_mm': 3.0,
                'base_time': 12
            },
            'turning': {
                'type': 'turning',
                'tool': 'CNMG Insert',
                'tool_diameter_mm': 0,  # Insert
                'doc_mm': 2.0,
                'base_time': 6
            },
            'threading': {
                'type': 'threading',
                'tool': 'Thread Mill',
                'tool_diameter_mm': 8,
                'doc_mm': 1.5,
                'base_time': 10
            }
        }
        
        op_template = operations_db.get(feature, operations_db['holes'])
        
        # Calculate speeds/feeds
        cutting_speed = material_props['cutting_speed_m_min']
        tool_dia = op_template['tool_diameter_mm']
        
        if tool_dia > 0:
            spindle_rpm = int((cutting_speed * 1000) / (math.pi * tool_dia))
            spindle_rpm = min(spindle_rpm, 4000)  # Max spindle speed
        else:
            spindle_rpm = 500  # Default for inserts
        
        feed_per_tooth = material_props['feed_per_tooth']
        num_flutes = 4  # Default
        feed_rate = spindle_rpm * num_flutes * feed_per_tooth
        
        # Estimate cycle time (affected by material hardness)
        hardness_factor = material_props['hardness_hb'] / 100
        cycle_time = op_template['base_time'] * hardness_factor
        
        return {
            'sequence_number': sequence,
            'operation_type': op_template['type'],
            'tool_description': op_template['tool'],
            'tool_diameter_mm': op_template['tool_diameter_mm'],
            'spindle_rpm': spindle_rpm,
            'feed_rate_mmpm': round(feed_rate, 1),
            'depth_of_cut_mm': op_template['doc_mm'],
            'cycle_time_minutes': round(cycle_time, 2),
            'coolant_type': 'flood' if material_props['hardness_hb'] > 150 else 'mist'
        }
    
    def _rate_complexity(self, part_type: str, num_operations: int) -> float:
        """Rate part complexity 1.0-5.0"""
        base_complexity = {
            'bearing_housing': 2.5,
            'shaft': 2.0,
            'bracket': 1.5,
            'pulley': 3.0,
            'gear': 4.5
        }
        
        complexity = base_complexity.get(part_type, 2.5)
        complexity += (num_operations - 3) * 0.2  # More operations = more complex
        
        return min(5.0, max(1.0, complexity))


class SyntheticOperationDataGenerator:
    """
    Generates realistic sensor data streams for machining operations
    """
    
    def __init__(self):
        self.material_db = MaterialDatabase()
    
    def generate_operation_stream(self, 
                                  operation: Dict,
                                  material: str,
                                  duration_minutes: float = 10.0,
                                  sample_rate_hz: int = 1000,
                                  inject_failures: bool = True) -> List[Dict]:
        """
        Generate time-series sensor data for operation
        
        Args:
            operation: Operation specification
            material: Material being machined
            duration_minutes: Duration of operation
            sample_rate_hz: Sampling frequency
            inject_failures: Whether to inject realistic failures
        
        Returns:
            List of sensor readings with timestamps
        """
        num_samples = int(duration_minutes * 60 * sample_rate_hz)
        material_props = self.material_db.MATERIALS[material]
        
        # Base operating conditions
        spindle_load_base = self._calculate_spindle_load(operation, material_props)
        temp_base = 25  # Room temperature
        
        data_stream = []
        tool_wear = 0.0
        
        for i in range(num_samples):
            time_seconds = i / sample_rate_hz
            timestamp = datetime.now() + timedelta(seconds=time_seconds)
            
            # Tool wear progression (Taylor tool life equation)
            tool_life_minutes = 120  # Example: 2 hours
            wear_rate = (time_seconds / 60) / tool_life_minutes
            tool_wear = min(100, wear_rate * 100)
            
            # Spindle load increases with tool wear
            wear_factor = 1 + (tool_wear / 100) * 0.4  # Up to 40% increase
            spindle_load = spindle_load_base * wear_factor + random.gauss(0, 2)
            spindle_load = max(0, min(100, spindle_load))
            
            # Vibration (composite of multiple frequencies)
            vib_x = self._generate_vibration(time_seconds, base_freq=150, amplitude=0.05)
            vib_y = self._generate_vibration(time_seconds, base_freq=180, amplitude=0.04)
            vib_z = self._generate_vibration(time_seconds, base_freq=200, amplitude=0.03)
            
            # Inject chatter if near resonance
            if inject_failures:
                chatter = self._check_chatter(operation['spindle_rpm'])
                if chatter['detected']:
                    vib_x += chatter['amplitude'] * random.random()
                    vib_y += chatter['amplitude'] * random.random()
                    vib_z += chatter['amplitude'] * 0.5 * random.random()
            
            # Temperature rises gradually
            temp_rise_rate = 0.25  # °C per minute
            temperature = temp_base + (time_seconds / 60) * temp_rise_rate + random.gauss(0, 0.5)
            
            # Dimensional accuracy degrades with wear
            nominal_accuracy = 0.01  # mm
            dimensional_error = nominal_accuracy * (1 + tool_wear / 100) + random.gauss(0, 0.002)
            
            # Surface finish degrades with wear
            target_ra = 1.6  # μm
            surface_roughness = target_ra * (1 +  tool_wear / 200) + random.gauss(0, 0.1)
            
            # Quality pass/fail
            tolerance = 0.05  # mm
            quality_pass = abs(dimensional_error) < tolerance
            
            data_point = {
                'timestamp': timestamp.isoformat(),
                'time_seconds': round(time_seconds, 3),
                'spindle_load_pct': round(spindle_load, 2),
                'vibration_x_mm': round(vib_x, 4),
                'vibration_y_mm': round(vib_y, 4),
                'vibration_z_mm': round(vib_z, 4),
                'temperature_c': round(temperature, 2),
                'tool_wear_pct': round(tool_wear, 2),
                'dimensional_accuracy_mm': round(abs(dimensional_error), 4),
                'surface_roughness_ra': round(surface_roughness, 3),
                'quality_pass': quality_pass,
                'chatter_detected': chatter['detected'] if inject_failures else False
            }
            
            data_stream.append(data_point)
        
        return data_stream
    
    def _calculate_spindle_load(self, operation: Dict, material_props: Dict) -> float:
        """Calculate base spindle load percentage"""
        # Simplified model
        hardness_factor = material_props['hardness_hb'] / 100
        doc_factor = operation['depth_of_cut_mm'] / 2.0
        
        base_load = 30 + (hardness_factor * 20) + (doc_factor * 15)
        return min(100, base_load)
    
    def _generate_vibration(self, time_sec: float, base_freq: float, amplitude: float) -> float:
        """Generate vibration signal"""
        # Multiple harmonics
        signal = 0
        for harmonic in [1, 2, 3]:
            freq = base_freq * harmonic
            phase = random.uniform(0, 2 * math.pi)
            signal += (amplitude / harmonic) * math.sin(2 * math.pi * freq * time_sec + phase)
        
        # Add noise
        signal += random.gauss(0, amplitude * 0.1)
        
        return signal
    
    def _check_chatter(self, spindle_rpm: int, natural_freq: int = 2500) -> Dict:
        """Check if spindle speed causes chatter"""
        # Chatter occurs near natural frequency harmonics
        harmonics = [natural_freq / i for i in range(1, 6)]
        
        for harmonic in harmonics:
            if abs(spindle_rpm - harmonic) < 100:
                severity = 1.0 - abs(spindle_rpm - harmonic) / 100
                return {
                    'detected': True,
                    'amplitude': 0.3 * severity,
                    'harmonic': harmonic
                }
        
        return {'detected': False, 'amplitude': 0, 'harmonic': None}


# Example usage
if __name__ == "__main__":
    # Generate synthetic part
    generator = PartTemplateGenerator()
    
    part = generator.generate_part('bearing_housing', vendor_requirements={'material': '6061-T6 Aluminum'})
    
    print("=" * 60)
    print("Generated Replacement Part:")
    print("=" * 60)
    print(json.dumps(part, indent=2))
    
    # Generate synthetic operation data
    op_generator = SyntheticOperationDataGenerator()
    
    operation = part['operations'][0]
    stream = op_generator.generate_operation_stream(
        operation=operation,
        material=part['material'],
        duration_minutes=1.0,  # 1 minute sample
        sample_rate_hz=10,  # 10 Hz for demo
        inject_failures=True
    )
    
    print("\n" + "=" * 60)
    print("Synthetic Operation Data (first 5 samples):")
    print("=" * 60)
    for sample in stream[:5]:
        print(json.dumps(sample, indent=2))
