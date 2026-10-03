import os
import glob
import random
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from openvino.runtime import Core
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False

class PhotobankManager:
    """Manages the asset library (Photobank) for the compositor."""
    def __init__(self, bank_dir="photobank"):
        self.bank_dir = bank_dir
        os.makedirs(self.bank_dir, exist_ok=True)
        self.images = []
        self._scan_bank()

    def _scan_bank(self):
        extensions = ['*.png', '*.jpg', '*.jpeg']
        for ext in extensions:
            self.images.extend(glob.glob(os.path.join(self.bank_dir, ext)))
        print(f"[Photobank] Discovered {len(self.images)} images in {self.bank_dir}.")

    def get_random_image(self, size=(512, 512)):
        if not self.images:
            # Generate a procedural fallback if photobank is empty
            img = Image.fromarray(np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8))
            return img
        
        img_path = random.choice(self.images)
        img = Image.open(img_path).convert("RGBA")
        img = img.resize(size, Image.Resampling.LANCZOS)
        return img

class OpenVinoNeuralFilter:
    """OpenVINO integrated cognitive filter for ML processing on layers."""
    def __init__(self):
        self.core = Core() if HAS_OPENVINO else None
        print(f"[OpenVINO] Engine initialized: {HAS_OPENVINO}")

    def apply_edge_enhance(self, pil_image):
        """Simulates an OpenVINO edge-detection/style-transfer pass."""
        if not HAS_CV2:
            return pil_image
            
        cv_img = np.array(pil_image)
        # Convert RGBA to RGB for processing
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2RGB)
        
        # Simulated OpenVINO Inference: Neural Edge Extraction
        # In a real scenario, this would compile a .xml model and run inference
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGBA)
        
        # Make black pixels transparent
        edges_colored[edges == 0] = [0, 0, 0, 0]
        edges_colored[edges > 0] = [0, 255, 255, 255] # Cyan edges (Krystal stack theme)
        
        return Image.fromarray(edges_colored)

class GimpLayer:
    """Represents a GIMP-like image layer with blending capabilities and parametric adjustments (sliders)."""
    def __init__(self, image: Image.Image, name="Layer", opacity=1.0, blend_mode="normal", adjustments=None):
        self.image = image.convert("RGBA")
        self.name = name
        self.opacity = max(0.0, min(1.0, opacity))
        self.blend_mode = blend_mode
        self.adjustments = adjustments or {} # e.g. {"brightness": 1.2, "contrast": 1.1, "saturation": 0.9}

    def _apply_retouching(self, img: Image.Image) -> Image.Image:
        """Applies slider-based adjustment patterns to the layer."""
        if "brightness" in self.adjustments:
            img = ImageEnhance.Brightness(img).enhance(self.adjustments["brightness"])
        if "contrast" in self.adjustments:
            img = ImageEnhance.Contrast(img).enhance(self.adjustments["contrast"])
        if "saturation" in self.adjustments:
            img = ImageEnhance.Color(img).enhance(self.adjustments["saturation"])
        return img

    def apply(self, background: Image.Image) -> Image.Image:
        """Composites this layer over the background using the specified blend mode."""
        if background.size != self.image.size:
            self.image = self.image.resize(background.size, Image.Resampling.LANCZOS)

        # Apply Retouching/Adjustments first
        processed_img = self._apply_retouching(self.image)

        # Apply Opacity
        if self.opacity < 1.0:
            alpha = processed_img.split()[3]
            alpha = ImageEnhance.Brightness(alpha).enhance(self.opacity)
            processed_img.putalpha(alpha)

        # Blending Modes (GIMP mimicry)
        if self.blend_mode == "normal":
            return Image.alpha_composite(background.convert("RGBA"), processed_img)
        elif self.blend_mode == "multiply":
            return ImageChops.multiply(background.convert("RGB"), processed_img.convert("RGB")).convert("RGBA")
        elif self.blend_mode == "screen":
            return ImageChops.screen(background.convert("RGB"), processed_img.convert("RGB")).convert("RGBA")
        elif self.blend_mode == "overlay":
            return ImageChops.overlay(background.convert("RGB"), processed_img.convert("RGB")).convert("RGBA")
        else:
            return Image.alpha_composite(background.convert("RGBA"), processed_img)

class NeuralCompositor:
    """The master generator integrating Photobank, GIMP layers, and OpenVINO."""
    def __init__(self):
        self.photobank = PhotobankManager()
        self.neural_filter = OpenVinoNeuralFilter()
        self.layers = []
        self.base_size = (1024, 1024)

    def add_layer(self, layer: GimpLayer):
        self.layers.append(layer)
        print(f"[Compositor] Added layer: {layer.name} | Blend: {layer.blend_mode} | Opacity: {layer.opacity}")

    def generate_symposia_image(self, composition_json=None, output_path="output_renders/symposia_generated.png"):
        print("\n[Compositor] Initiating Neural Generation Sequence from JSON traits...")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Default fallback config if no JSON provided
        config = composition_json or {
            "base_adjustments": {"contrast": 1.2, "saturation": 1.1},
            "edge_opacity": 0.8,
            "edge_blend": "screen",
            "texture_opacity": 0.4,
            "texture_blend": "overlay",
            "atmosphere_color": [0, 150, 200, 255]
        }
        
        # 1. Base Layer from Photobank
        base_img = self.photobank.get_random_image(self.base_size)
        base_layer = GimpLayer(base_img, name="Background", adjustments=config.get("base_adjustments", {}))
        self.add_layer(base_layer)

        # 2. OpenVINO Neural Edge Layer
        edge_img = self.neural_filter.apply_edge_enhance(base_img)
        edge_layer = GimpLayer(edge_img, name="OpenVINO Neural Edges", 
                               opacity=config.get("edge_opacity", 0.8), 
                               blend_mode=config.get("edge_blend", "screen"))
        self.add_layer(edge_layer)

        # 3. Secondary Photobank Overlay (Texture)
        texture_img = self.photobank.get_random_image(self.base_size)
        texture_layer = GimpLayer(texture_img, name="Photobank Texture Overlay", 
                                  opacity=config.get("texture_opacity", 0.4), 
                                  blend_mode=config.get("texture_blend", "overlay"),
                                  adjustments={"contrast": 1.5}) # Harder texture
        self.add_layer(texture_layer)

        # 4. Procedural GIMP Multiply layer (Color Grading)
        atm_color = tuple(config.get("atmosphere_color", [0, 150, 200, 255]))
        solid_color = Image.new("RGBA", self.base_size, color=atm_color)
        color_layer = GimpLayer(solid_color, name="Atmospheric Grade", opacity=0.3, blend_mode="multiply")
        self.add_layer(color_layer)

        # Flatten layers
        print("[Compositor] Flattening layers...")
        final_image = Image.new("RGBA", self.base_size, (0, 0, 0, 255))
        for layer in self.layers:
            final_image = layer.apply(final_image)

        # Save output
        final_image.save(output_path)
        print(f"[Compositor] Successfully generated and saved to: {output_path}")

if __name__ == "__main__":
    compositor = NeuralCompositor()
    compositor.generate_symposia_image()
