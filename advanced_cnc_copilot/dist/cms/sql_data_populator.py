"""
SQL Database Populator for Synthetic Manufacturing Data
Generates SQL INSERT statements from synthetic part/operation data
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List
import random

from cms.synthetic_data_generator import PartTemplateGenerator, SyntheticOperationDataGenerator


class SQLGenerator:
    """
    Generates SQL INSERT statements from synthetic data
    """
    
    def __init__(self, database_name: str = 'manufacturing_db'):
        self.db_name = database_name
        self.vendor_id_map = {}
        self.part_id_counter = 1
        self.operation_id_counter = 1
    
    def generate_complete_database(self, num_parts_per_type: int = 10) -> str:
        """
        Generate complete SQL script to populate database
        
        Args:
            num_parts_per_type: Number of parts to generate for each type
        
        Returns:
            Complete SQL script as string
        """
        sql_script = []
        
        # Header
        sql_script.append(f"-- Auto-generated Synthetic Manufacturing Database")
        sql_script.append(f"-- Generated: {datetime.now().isoformat()}")
        sql_script.append(f"-- Total parts: {num_parts_per_type * 5}")
        sql_script.append(f"\nUSE {self.db_name};\n")
        
        # Disable foreign key checks for insertion
        sql_script.append("SET FOREIGN_KEY_CHECKS = 0;\n")
        
        # 1. Generate vendors
        sql_script.append(self._generate_vendors())
        
        # 2. Generate part categories
        sql_script.append(self._generate_categories())
        
        # 3. Generate tools
        sql_script.append(self._generate_tools())
        
        # 4. Generate parts and operations
        part_generator = PartTemplateGenerator()
        
        part_types = ['bearing_housing', 'shaft', 'bracket', 'pulley', 'gear']
        
        for part_type in part_types:
            for i in range(num_parts_per_type):
                # Generate part
                part_data = part_generator.generate_part(part_type)
                
                # Generate SQL for part
                sql_script.append(self._generate_part_sql(part_data))
                
                # Generate SQL for operations
                for operation in part_data['operations']:
                    sql_script.append(self._generate_operation_sql(
                        self.part_id_counter, operation
                    ))
                    self.operation_id_counter += 1
                
                # Generate synthetic operation data
                sql_script.append(self._generate_synthetic_data_sql(
                    part_data, sample_rate=100  # 100 samples per operation
                ))
                
                self.part_id_counter += 1
        
        # 5. Generate orders
        sql_script.append(self._generate_orders(num_orders=20))
        
        # Re-enable foreign key checks
        sql_script.append("\nSET FOREIGN_KEY_CHECKS = 1;")
        
        return "\n".join(sql_script)
    
    def _generate_vendors(self) -> str:
        """Generate vendor data"""
        vendors = [
            {
                'name': 'Automotive Parts Inc.',
                'industry': 'automotive',
                'location': 'Detroit, MI',
                'quality_rating': 4.5,
                'min_order_qty': 100,
                'lead_time_days': 14
            },
            {
                'name': 'Aerospace Components Ltd.',
                'industry': 'aerospace',
                'location': 'Seattle, WA',
                'quality_rating': 4.9,
                'min_order_qty': 10,
                'lead_time_days': 30
            },
            {
                'name': 'Medical Device Solutions',
                'industry': 'medical',
                'location': 'Boston, MA',
                'quality_rating': 4.8,
                'min_order_qty': 50,
                'lead_time_days': 21
            },
            {
                'name': 'Industrial Equipment Co.',
                'industry': 'industrial',
                'location': 'Chicago, IL',
                'quality_rating': 4.3,
                'min_order_qty': 200,
                'lead_time_days': 10
            },
            {
                'name': 'Marine Systems Manufacturing',
                'industry': 'marine',
                'location': 'San Diego, CA',
                'quality_rating': 4.6,
                'min_order_qty': 25,
                'lead_time_days': 20
            }
        ]
        
        sql = ["\n-- Vendors"]
        for idx, vendor in enumerate(vendors, 1):
            self.vendor_id_map[vendor['name']] = idx
            sql.append(f"""
INSERT INTO vendors (vendor_id, name, industry, location, quality_rating, min_order_quantity, lead_time_days)
VALUES ({idx}, '{vendor['name']}', '{vendor['industry']}', '{vendor['location']}', 
        {vendor['quality_rating']}, {vendor['min_order_qty']}, {vendor['lead_time_days']});
""")
        
        return "\n".join(sql)
    
    def _generate_categories(self) -> str:
        """Generate part categories"""
        categories = [
            {'id': 1, 'name': 'Bearings & Housings', 'complexity': 2.5},
            {'id': 2, 'name': 'Shafts & Spindles', 'complexity': 2.0},
            {'id': 3, 'name': 'Brackets & Mounts', 'complexity': 1.5},
            {'id': 4, 'name': 'Pulleys & Sheaves', 'complexity': 3.0},
            {'id': 5, 'name': 'Gears & Sprockets', 'complexity': 4.5}
        ]
        
        sql = ["\n-- Part Categories"]
        for cat in categories:
            sql.append(f"""
INSERT INTO part_categories (category_id, name, complexity_factor)
VALUES ({cat['id']}, '{cat['name']}', {cat['complexity']});
""")
        
        return "\n".join(sql)
    
    def _generate_tools(self) -> str:
        """Generate tool library"""
        tools = [
            {'number': 'T01', 'type': 'Face Mill', 'diameter': 100, 'flutes': 8, 'coating': 'TiAlN'},
            {'number': 'T02', 'type': 'End Mill', 'diameter': 12, 'flutes': 4, 'coating': 'TiCN'},
            {'number': 'T03', 'type': 'Drill', 'diameter': 10, 'flutes': 2, 'coating': 'TiN'},
            {'number': 'T04', 'type': 'Boring Bar', 'diameter': 25, 'flutes': 1, 'coating': 'Carbide'},
            {'number': 'T05', 'type': 'Thread Mill', 'diameter': 8, 'flutes': 3, 'coating': 'AlTiN'}
        ]
        
        sql = ["\n-- Tools"]
        for tool in tools:
            sql.append(f"""
INSERT INTO tools (tool_number, tool_type, diameter_mm, flutes, coating, max_rpm, cost_usd)
VALUES ('{tool['number']}', '{tool['type']}', {tool['diameter']}, {tool['flutes']}, 
        '{tool['coating']}', 4000, {random.uniform(50, 500):.2f});
""")
        
        return "\n".join(sql)
    
    def _generate_part_sql(self, part_data: Dict) -> str:
        """Generate SQL for single part"""
        category_map = {
            'bearing_housing': 1,
            'shaft': 2,
            'bracket': 3,
            'pulley': 4,
            'gear': 5
        }
        
        category_id = category_map.get(part_data['part_type'], 1)
        
        # Escape single quotes in strings
        name = part_data['name'].replace("'", "''")
        description = part_data['description'].replace("'", "''")
        
        sql = f"""
-- Part: {part_data['part_number']}
INSERT INTO parts (
    part_id, part_number, name, description, category_id,
    original_manufacturer, original_part_number, end_of_life_reason,
    material, weight_kg, dimensions, tolerances, surface_finish,
    estimated_cycle_time_minutes, estimated_cost_usd, complexity_score
) VALUES (
    {self.part_id_counter},
    '{part_data['part_number']}',
    '{name}',
    '{description}',
    {category_id},
    'Original Equipment Manufacturer',
    'OEM-{random.randint(10000, 99999)}',
    '{part_data['end_of_life_reason']}',
    '{part_data['material']}',
    {part_data['weight_kg']},
    '{json.dumps(part_data['dimensions'])}',
    '{json.dumps(part_data['tolerances'])}',
    '{part_data['surface_finish']}',
    {part_data['estimated_cycle_time_minutes']},
    {part_data['estimated_cost_usd']},
    {part_data['complexity_score']}
);
"""
        return sql
    
    def _generate_operation_sql(self, part_id: int, operation: Dict) -> str:
        """Generate SQL for machining operation"""
        sql = f"""
INSERT INTO operations (
    operation_id, part_id, sequence_number, operation_type,
    tool_description, tool_diameter_mm,
    spindle_rpm, feed_rate_mmpm, depth_of_cut_mm,
    cycle_time_minutes, coolant_type
) VALUES (
    {self.operation_id_counter},
    {part_id},
    {operation['sequence_number']},
    '{operation['operation_type']}',
    '{operation['tool_description']}',
    {operation['tool_diameter_mm']},
    {operation['spindle_rpm']},
    {operation['feed_rate_mmpm']},
    {operation['depth_of_cut_mm']},
    {operation['cycle_time_minutes']},
    '{operation['coolant_type']}'
);
"""
        return sql
    
    def _generate_synthetic_data_sql(self, part_data: Dict, sample_rate: int = 100) -> str:
        """Generate synthetic operation data"""
        op_generator = SyntheticOperationDataGenerator()
        
        sql_statements = ["\n-- Synthetic operation data"]
        
        for operation in part_data['operations']:
            # Generate sensor stream
            stream = op_generator.generate_operation_stream(
                operation=operation,
                material=part_data['material'],
                duration_minutes=operation['cycle_time_minutes'],
                sample_rate_hz=1,  # 1 Hz for SQL (reduce data volume)
                inject_failures=True
            )
            
            # Sample the stream
            step = max(1, len(stream) // sample_rate)
            sampled_stream = stream[::step][:sample_rate]
            
            for sample in sampled_stream:
                sql_statements.append(f"""
INSERT INTO synthetic_operations (
    operation_id, simulation_timestamp,
    spindle_load_pct, vibration_x, vibration_y, vibration_z,
    temperature_c, tool_wear_pct, dimensional_accuracy_mm,
    surface_roughness_ra, chatter_detected, quality_pass
) VALUES (
    {self.operation_id_counter},
    '{sample['timestamp']}',
    {sample['spindle_load_pct']},
    {sample['vibration_x_mm']},
    {sample['vibration_y_mm']},
    {sample['vibration_z_mm']},
    {sample['temperature_c']},
    {sample['tool_wear_pct']},
    {sample['dimensional_accuracy_mm']},
    {sample['surface_roughness_ra']},
    {1 if sample['chatter_detected'] else 0},
    {1 if sample['quality_pass'] else 0}
);
""")
        
        return "\n".join(sql_statements)
    
    def _generate_orders(self, num_orders: int = 20) -> str:
        """Generate sample orders"""
        sql = ["\n-- Sample Orders"]
        
        for order_id in range(1, num_orders + 1):
            vendor_id = random.randint(1, 5)
            order_date = datetime.now() - timedelta(days=random.randint(1, 90))
            required_date = order_date + timedelta(days=random.randint(14, 45))
            status = random.choice(['quoted', 'ordered', 'in_production', 'completed'])
            
            sql.append(f"""
INSERT INTO orders (order_id, vendor_id, order_date, required_date, status, total_amount_usd)
VALUES ({order_id}, {vendor_id}, '{order_date.date()}', '{required_date.date()}', 
        '{status}', {random.uniform(1000, 50000):.2f});
""")
            
            # Add 1-5 items per order
            num_items = random.randint(1, 5)
            for item_num in range(1, num_items + 1):
                part_id = random.randint(1, min(self.part_id_counter - 1, 50))
                quantity = random.choice([1, 5, 10, 25, 50, 100])
                
                sql.append(f"""
INSERT INTO order_items (order_id, part_id, quantity, unit_price_usd, quantity_produced, quantity_passed)
VALUES ({order_id}, {part_id}, {quantity}, {random.uniform(10, 500):.2f}, 
        {random.randint(0, quantity)}, {random.randint(0, quantity)});
""")
        
        return "\n".join(sql)


# Main execution
if __name__ == "__main__":
    print("Generating synthetic manufacturing database...")
    
    generator = SQLGenerator('manufacturing_db')
    
    # Generate complete SQL script
    sql_script = generator.generate_complete_database(num_parts_per_type=5)
    
    # Save to file
    output_file = 'synthetic_manufacturing_data.sql'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(sql_script)
    
    print(f"✅ Generated {output_file}")
    print(f"   Parts created: {generator.part_id_counter - 1}")
    print(f"   Operations created: {generator.operation_id_counter - 1}")
    print("\nTo load into MySQL:")
    print(f"   mysql -u root -p < {output_file}")
