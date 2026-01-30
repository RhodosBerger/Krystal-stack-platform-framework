"""
Database Import Script for Manufacturing Knowledge
Loads scraped knowledge base into PostgreSQL

IMPORTS:
- Best practices → New 'best_practices' table
- Part templates → 'parts' table
- Speeds/feeds → Part metadata
- Material properties → 'materials' table

USAGE:
python scripts/load_knowledge_to_db.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection import DatabaseConnectionManager
from database.repository import (
    MaterialRepository,
    PartRepository,
    ProducerRepository
)


def load_knowledge_base(filename: str = 'manufacturing_knowledge_base.json') -> dict:
    """Load knowledge base from JSON file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Knowledge base file not found: {filename}")
        print("   Run: python scripts/import_manufacturing_knowledge.py first")
        return None


def create_best_practices_table(db_manager):
    """Create best_practices table if it doesn't exist"""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS best_practices (
        id SERIAL PRIMARY KEY,
        category VARCHAR(50) NOT NULL,
        material VARCHAR(50),
        operation VARCHAR(50),
        rule TEXT NOT NULL,
        source VARCHAR(200),
        confidence DECIMAL(3,2),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_best_practices_category ON best_practices(category);
    CREATE INDEX IF NOT EXISTS idx_best_practices_material ON best_practices(material);
    """
    
    with db_manager.session_scope() as session:
        session.execute(create_table_sql)
        session.commit()
    
    print("✅ Best practices table ready")


def import_materials(knowledge_base, db_manager):
    """Import materials with speeds/feeds properties"""
    print("\n📦 Importing Materials...")
    
    materials_data = knowledge_base.get('speeds_feeds', {})
    imported_count = 0
    
    with db_manager.session_scope() as session:
        for material_name, operations in materials_data.items():
            # Check if material exists
            existing = MaterialRepository.get_by_name(session, material_name)
            
            if existing:
                print(f"   ⏭️  {material_name} already exists, skipping")
                continue
            
            # Extract properties from roughing operation
            roughing = operations.get('roughing', {})
            finishing = operations.get('finishing', {})
            
            # Create material properties
            properties = {
                'cutting_speed_roughing_m_min': roughing.get('cutting_speed_m_min'),
                'cutting_speed_finishing_m_min': finishing.get('cutting_speed_m_min'),
                'feed_per_tooth_roughing_mm': roughing.get('feed_per_tooth_mm'),
                'feed_per_tooth_finishing_mm': finishing.get('feed_per_tooth_mm'),
                'depth_of_cut_roughing_mm': roughing.get('depth_of_cut_mm'),
                'depth_of_cut_finishing_mm': finishing.get('depth_of_cut_mm'),
                'tool_type_roughing': roughing.get('tool_type'),
                'tool_type_finishing': finishing.get('tool_type')
            }
            
            # Create material
            material = MaterialRepository.create(
                session,
                name=material_name,
                category='Metal' if 'Steel' in material_name or 'Aluminum' in material_name or 'Titanium' in material_name else 'Other',
                properties=properties
            )
            
            print(f"   ✅ Imported {material_name}")
            imported_count += 1
    
    print(f"✅ Imported {imported_count} materials")
    return imported_count


def import_best_practices(knowledge_base, db_manager):
    """Import best practices"""
    print("\n📚 Importing Best Practices...")
    
    practices = knowledge_base.get('best_practices', [])
    imported_count = 0
    
    with db_manager.session_scope() as session:
        for practice in practices:
            # Insert best practice
            insert_sql = """
            INSERT INTO best_practices (category, material, operation, rule, source, confidence)
            VALUES (:category, :material, :operation, :rule, :source, :confidence)
            """
            
            session.execute(insert_sql, {
                'category': practice['category'],
                'material': practice['material'],
                'operation': practice['operation'],
                'rule': practice['rule'],
                'source': practice['source'],
                'confidence': practice['confidence']
            })
            
            imported_count += 1
        
        session.commit()
    
    print(f"✅ Imported {imported_count} best practices")
    return imported_count


def import_parts_library(knowledge_base, db_manager):
    """Import parts library"""
    print("\n🔧 Importing Parts Library...")
    
    parts = knowledge_base.get('parts_library', [])
    imported_count = 0
    
    with db_manager.session_scope() as session:
        for part_data in parts:
            # Get or create material
            material = MaterialRepository.get_by_name(session, part_data['material'])
            if not material:
                print(f"   ⚠️  Material {part_data['material']} not found, skipping {part_data['description']}")
                continue
            
            # Create part
            part = PartRepository.create(
                session,
                name=part_data['description'],
                part_number=f"STD-{part_data['part_type'].upper()}-{imported_count+1:04d}",
                material_id=material.id,
                dimensions=part_data['dimensions'],
                properties={
                    'template': part_data['template'],
                    'part_type': part_data['part_type']
                }
            )
            
            print(f"   ✅ Imported {part_data['description']}")
            imported_count += 1
    
    print(f"✅ Imported {imported_count} parts")
    return imported_count


def generate_import_summary(stats: dict):
    """Generate import summary"""
    print("\n" + "=" * 70)
    print("IMPORT SUMMARY")
    print("=" * 70)
    print(f"""
📊 Import Statistics:
   Materials:       {stats['materials']} imported
   Best Practices:  {stats['best_practices']} imported
   Parts Library:   {stats['parts']} imported
   
✅ Database successfully populated with manufacturing knowledge!

📚 Knowledge Base Contents:
   - Speeds & Feeds for {stats['materials']} materials
   - {stats['best_practices']} expert best practices
   - {stats['parts']} standard part templates
   
🎯 Next Steps:
   1. Query best practices: SELECT * FROM best_practices WHERE category='speeds_feeds';
   2. View materials: SELECT * FROM materials;
   3. Browse parts: SELECT * FROM parts;
   
🚀 AI systems can now learn from this curated knowledge!
""")


def main():
    """Main import function"""
    print("=" * 70)
    print("Manufacturing Knowledge Database Import")
    print("=" * 70)
    
    # Load knowledge base
    knowledge_base = load_knowledge_base()
    if not knowledge_base:
        return
    
    print(f"\n📖 Loaded knowledge base from JSON")
    print(f"   Best practices: {len(knowledge_base['best_practices'])}")
    print(f"   Parts library: {len(knowledge_base['parts_library'])}")
    print(f"   Materials: {len(knowledge_base['speeds_feeds'])}")
    
    # Initialize database connection
    print("\n🔌 Connecting to database...")
    db_manager = DatabaseConnectionManager()
    
    # Create best practices table
    create_best_practices_table(db_manager)
    
    # Import data
    stats = {}
    
    try:
        stats['materials'] = import_materials(knowledge_base, db_manager)
        stats['best_practices'] = import_best_practices(knowledge_base, db_manager)
        stats['parts'] = import_parts_library(knowledge_base, db_manager)
        
        # Generate summary
        generate_import_summary(stats)
    
    except Exception as e:
        print(f"\n❌ Error during import: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
