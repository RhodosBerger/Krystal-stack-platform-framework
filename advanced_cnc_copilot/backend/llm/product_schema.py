"""
Product Schema and Registry 📦
Responsibility:
1. Define searchable Product schema.
2. Manage product database with search capabilities.
"""
import uuid
from typing import List, Dict, Any, Optional

class Product:
    def __init__(self, name: str, category: str, dim_x: float = 100, dim_y: float = 50, dim_z: float = 25, tags: List[str] = None):
        self.id = f"PROD-{uuid.uuid4().hex[:6].upper()}"
        self.name = name
        self.category = category
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.tags = tags or []
        self.status = "ACTIVE"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "dim_x": self.dim_x,
            "dim_y": self.dim_y,
            "dim_z": self.dim_z,
            "tags": self.tags,
            "status": self.status
        }

class ProductRegistry:
    def __init__(self):
        self._products: Dict[str, Product] = {}
        # Seed some demo products
        self._seed_demo_products()

    def _seed_demo_products(self):
        demos = [
            Product("Engine Bracket V3", "Automotive", 120, 80, 35, ["precision", "aerospace"]),
            Product("Gear Blank Assembly", "Mechanical", 60, 60, 20, ["gear", "transmission"]),
            Product("Heat Sink Panel", "Electronics", 150, 100, 10, ["thermal", "cooling"]),
            Product("Structural Frame A1", "Industrial", 300, 200, 50, ["heavy", "structural"]),
        ]
        for p in demos:
            self._products[p.id] = p

    def add(self, product: Product) -> str:
        self._products[product.id] = product
        return product.id

    def get(self, product_id: str) -> Optional[Product]:
        return self._products.get(product_id)

    def list_all(self) -> List[Product]:
        return list(self._products.values())

    def search(self, query: str) -> List[Product]:
        """
        Searches products by name, category, or tags.
        """
        query_lower = query.lower()
        results = []
        for p in self._products.values():
            if query_lower in p.name.lower() or \
               query_lower in p.category.lower() or \
               any(query_lower in tag.lower() for tag in p.tags):
                results.append(p)
        return results

# Global Instance
product_registry = ProductRegistry()
