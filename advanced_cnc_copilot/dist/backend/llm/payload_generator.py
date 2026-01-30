"""
LLM-Assisted Payload Generator 🤖
Responsibility:
1. Generate multiple manufacturing payloads based on product definitions.
2. Simulate LLM assistance for intelligent payload construction.
3. Support batch payload generation.
"""
import uuid
from typing import List, Dict, Any

class PayloadType:
    GCODE = "GCODE"
    JSON = "JSON"
    XML = "XML"
    MARKDOWN = "MARKDOWN"

class LLMPayloadGenerator:
    def __init__(self):
        self.generation_log = []

    def generate_payloads(self, product: Dict[str, Any], payload_types: List[str], llm_prompt: str = "") -> Dict[str, Any]:
        """
        Generates multiple payloads for a defined product using LLM-style assistance.
        """
        payloads = []
        batch_id = f"BATCH-{uuid.uuid4().hex[:6].upper()}"
        
        for ptype in payload_types:
            payload = self._generate_single_payload(product, ptype, llm_prompt)
            payloads.append(payload)
        
        result = {
            "batch_id": batch_id,
            "product_id": product.get("id", "UNKNOWN"),
            "product_name": product.get("name", "Untitled"),
            "payloads": payloads,
            "llm_context": llm_prompt or "Default manufacturing context"
        }
        
        self.generation_log.append(result)
        return result

    def _generate_single_payload(self, product: Dict[str, Any], ptype: str, llm_prompt: str) -> Dict[str, Any]:
        """
        Generates a single payload based on type.
        """
        payload_id = f"PL-{uuid.uuid4().hex[:8].upper()}"
        content = ""
        
        if ptype == PayloadType.GCODE:
            content = self._build_gcode(product, llm_prompt)
        elif ptype == PayloadType.JSON:
            content = self._build_json(product, llm_prompt)
        elif ptype == PayloadType.XML:
            content = self._build_xml(product, llm_prompt)
        elif ptype == PayloadType.MARKDOWN:
            content = self._build_markdown(product, llm_prompt)
        
        return {
            "id": payload_id,
            "type": ptype,
            "content": content,
            "size_bytes": len(content)
        }

    def _build_gcode(self, product: Dict[str, Any], prompt: str) -> str:
        return f"""%
(PRODUCT: {product.get('name', 'Unknown')})
(LLM_CONTEXT: {prompt[:50] if prompt else 'Standard'})
G21 G90
G01 X{product.get('dim_x', 100)} Y{product.get('dim_y', 50)} Z{product.get('dim_z', 25)} F500
M30
%"""

    def _build_json(self, product: Dict[str, Any], prompt: str) -> str:
        import json
        return json.dumps({
            "product": product,
            "llm_context": prompt,
            "manufacturing_ready": True
        }, indent=2)

    def _build_xml(self, product: Dict[str, Any], prompt: str) -> str:
        return f"""<?xml version="1.0"?>
<ManufacturingPayload>
    <Product id="{product.get('id', 'N/A')}" name="{product.get('name', 'Unknown')}" />
    <LLMContext>{prompt or 'Default'}</LLMContext>
    <Status>READY</Status>
</ManufacturingPayload>"""

    def _build_markdown(self, product: Dict[str, Any], prompt: str) -> str:
        return f"""# Manufacturing Report: {product.get('name', 'Unknown')}

## Product Details
- **ID:** {product.get('id', 'N/A')}
- **Dimensions:** {product.get('dim_x', 0)} x {product.get('dim_y', 0)} x {product.get('dim_z', 0)} mm

## LLM Context
> {prompt or 'No specific context provided.'}

## Status
✅ Ready for Production
"""

# Global Instance
llm_payload_generator = LLMPayloadGenerator()
