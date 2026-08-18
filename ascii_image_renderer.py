#!/usr/bin/env python3
"""
ASCII Image Renderer for GAMESA Framework

This module implements an ASCII image renderer that can convert images to ASCII art
and integrate with the hexadecimal system and existing framework components.
"""

import numpy as np
from PIL import Image
import math
from typing import Tuple, List, Optional
from dataclasses import dataclass
import base64
from io import BytesIO


@dataclass
class ASCIIConfig:
    """Configuration for ASCII rendering."""
    width: int = 80
    height: int = 40
    charset: str = " .:-=+*#%@"
    invert: bool = False
    scale_factor: float = 0.43  # Adjust for terminal character aspect ratio


class ASCIIImageRenderer:
    """
    ASCII Image Renderer that converts images to ASCII art.
    
    Can be integrated with the hexadecimal system and composition generator.
    """
    
    def __init__(self, config: ASCIIConfig = None):
        self.config = config or ASCIIConfig()
    
    def resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image to ASCII dimensions."""
        width, height = self.config.width, self.config.height
        # Adjust height to account for character aspect ratio
        adjusted_height = int(height / self.config.scale_factor)
        
        return image.resize((width, adjusted_height))
    
    def grayscale_image(self, image: Image.Image) -> Image.Image:
        """Convert image to grayscale."""
        if image.mode != 'L':
            image = image.convert('L')
        return image
    
    def pixels_to_ascii(self, image: Image.Image) -> str:
        """Convert grayscale image pixels to ASCII characters."""
        pixels = np.array(image)
        ascii_chars = self.config.charset
        
        # Normalize pixel values to range [0, len(ascii_chars)-1]
        normalized_pixels = pixels / 255.0
        if self.config.invert:
            normalized_pixels = 1.0 - normalized_pixels
        
        # Map normalized values to ASCII characters
        ascii_indices = (normalized_pixels * (len(ascii_chars) - 1)).astype(int)
        ascii_indices = np.clip(ascii_indices, 0, len(ascii_chars) - 1)
        
        # Convert to ASCII art
        ascii_art = ""
        for row in ascii_indices:
            for idx in row:
                ascii_art += ascii_chars[idx]
            ascii_art += "\n"
        
        return ascii_art
    
    def render_image(self, image: Image.Image, config: ASCIIConfig = None) -> str:
        """Render an image as ASCII art."""
        if config:
            original_config = self.config
            self.config = config
        else:
            original_config = None
        
        try:
            # Resize image
            resized = self.resize_image(image)
            
            # Convert to grayscale
            grayscale = self.grayscale_image(resized)
            
            # Convert to ASCII
            ascii_art = self.pixels_to_ascii(grayscale)
            
            return ascii_art
        finally:
            if original_config:
                self.config = original_config
    
    def render_from_path(self, image_path: str, config: ASCIIConfig = None) -> str:
        """Render an image from file path as ASCII art."""
        image = Image.open(image_path)
        return self.render_image(image, config)
    
    def render_from_bytes(self, image_bytes: bytes, config: ASCIIConfig = None) -> str:
        """Render an image from bytes as ASCII art."""
        image = Image.open(BytesIO(image_bytes))
        return self.render_image(image, config)
    
    def render_from_url(self, image_url: str, config: ASCIIConfig = None) -> str:
        """Render an image from URL as ASCII art (requires requests)."""
        try:
            import requests
            response = requests.get(image_url)
            response.raise_for_status()
            return self.render_from_bytes(response.content, config)
        except ImportError:
            raise ImportError("requests library required for URL rendering")
        except Exception as e:
            raise e


class HexadecimalASCIIConverter:
    """
    Converts hexadecimal data to ASCII representations.
    
    Integrates with the hexadecimal system to provide visual representations
    of hex data in ASCII format.
    """
    
    def __init__(self):
        self.renderer = ASCIIImageRenderer()
    
    def hex_to_ascii_art(self, hex_data: str, width: int = 16) -> str:
        """
        Convert hexadecimal string to ASCII art representation.
        
        Args:
            hex_data: Hexadecimal string to convert
            width: Number of hex values per line
            
        Returns:
            ASCII art representation of the hex data
        """
        # Remove any spaces or formatting
        hex_clean = hex_data.replace(' ', '').replace('0x', '').upper()
        
        # Ensure even length
        if len(hex_clean) % 2 != 0:
            hex_clean = '0' + hex_clean
        
        # Split into bytes
        bytes_list = [hex_clean[i:i+2] for i in range(0, len(hex_clean), 2)]
        
        # Format as hex dump
        ascii_art = ""
        for i in range(0, len(bytes_list), width):
            line_bytes = bytes_list[i:i+width]
            
            # Create hex portion
            hex_part = ' '.join(line_bytes).ljust(width * 3)
            
            # Create ASCII portion
            ascii_part = ""
            for hex_byte in line_bytes:
                try:
                    byte_val = int(hex_byte, 16)
                    if 32 <= byte_val <= 126:  # Printable ASCII range
                        ascii_part += chr(byte_val)
                    else:
                        ascii_part += '.'
                except ValueError:
                    ascii_part += '.'
            
            ascii_art += f"{i:08X}: {hex_part} |{ascii_part}|\n"
        
        return ascii_art
    
    def visualize_hex_distribution(self, hex_values: List[int], width: int = 60, height: int = 20) -> str:
        """
        Create an ASCII visualization of hex value distribution.
        
        Args:
            hex_values: List of hex values (0-255)
            width: Width of the visualization
            height: Height of the visualization
            
        Returns:
            ASCII art visualization of the distribution
        """
        if not hex_values:
            return "No data to visualize"
        
        # Create histogram
        histogram = [0] * 256
        for val in hex_values:
            if 0 <= val <= 255:
                histogram[val] += 1
        
        # Normalize histogram
        max_count = max(histogram)
        if max_count == 0:
            return "No data to visualize"
        
        # Create visualization grid
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Scale factors
        x_scale = 256 / width
        y_scale = max_count / height if max_count > 0 else 1
        
        # Plot the histogram
        for x in range(width):
            start_idx = int(x * x_scale)
            end_idx = int((x + 1) * x_scale)
            if end_idx > 256:
                end_idx = 256
            
            # Find max value in this range
            max_val = 0
            for i in range(start_idx, end_idx):
                if i < len(histogram) and histogram[i] > max_val:
                    max_val = histogram[i]
            
            # Calculate height for this column
            bar_height = int((max_val / y_scale) if y_scale > 0 else 0)
            bar_height = min(bar_height, height)
            
            # Draw the bar
            for y in range(height - bar_height, height):
                grid[y][x] = '#'
        
        # Convert grid to string
        visualization = ""
        for row in grid:
            visualization += ''.join(row) + '\n'
        
        return visualization
    
    def hex_composition_to_ascii(self, composition_data: dict) -> str:
        """
        Convert composition data to ASCII representation.
        
        Args:
            composition_data: Composition data from the composition generator
            
        Returns:
            ASCII art representation of the composition
        """
        if not composition_data:
            return "No composition data"
        
        lines = []
        lines.append("COMPOSITION VISUALIZATION")
        lines.append("=" * 30)
        
        # Show composition ID
        if 'composition_id' in composition_data:
            lines.append(f"ID: {composition_data['composition_id']}")
        
        # Show efficiency and risk
        if 'efficiency_score' in composition_data:
            eff = composition_data['efficiency_score']
            lines.append(f"Efficiency: {eff:.2f}")
        
        if 'risk_level' in composition_data:
            lines.append(f"Risk: {composition_data['risk_level'].upper()}")
        
        # Show hex values as a visual pattern
        if 'hex_values' in composition_data:
            hex_vals = composition_data['hex_values']
            lines.append(f"Hex Values: {len(hex_vals)} items")
            
            # Create a simple visual pattern
            pattern = self._create_hex_pattern(hex_vals)
            lines.append(pattern)
        
        # Show resource allocation
        if 'resources' in composition_data:
            lines.append("RESOURCES:")
            for res_type, res_data in composition_data['resources'].items():
                if isinstance(res_data, dict):
                    hex_val = res_data.get('hex_value', 'N/A')
                    alloc = res_data.get('allocation', 'N/A')
                    priority = res_data.get('priority', 'N/A')
                    lines.append(f"  {res_type}: 0x{hex_val:02X}, Alloc: {alloc:.1f}, Prio: {priority}")
        
        return '\n'.join(lines)
    
    def _create_hex_pattern(self, hex_values: List[int], width: int = 20) -> str:
        """Create a visual pattern from hex values."""
        if not hex_values:
            return ""
        
        # Normalize values to 0-9 for pattern
        normalized = [(v % 10) for v in hex_values]
        
        pattern = "PATTERN: "
        for val in normalized:
            pattern += str(val)
        
        # Create a simple bar visualization
        max_val = max(normalized) if normalized else 0
        if max_val == 0:
            return pattern
        
        bar_lines = []
        for level in range(max_val, 0, -1):
            line = "         "  # Indent
            for val in normalized:
                if val >= level:
                    line += "# "
                else:
                    line += "  "
            bar_lines.append(line)
        
        return pattern + "\n" + '\n'.join(bar_lines)


def demo_ascii_image_rendering():
    """Demonstrate ASCII image rendering capabilities."""
    print("=" * 80)
    print("ASCII IMAGE RENDERING DEMONSTRATION")
    print("=" * 80)
    
    # Create hex ASCII converter
    hex_converter = HexadecimalASCIIConverter()
    print("[OK] Hexadecimal ASCII converter initialized")
    
    # Demonstrate hex to ASCII art
    sample_hex = "48656C6C6F20576F726C64"  # "Hello World" in hex
    hex_art = hex_converter.hex_to_ascii_art(sample_hex)
    print(f"\nHex to ASCII Art for 'Hello World':")
    print(hex_art)
    
    # Demonstrate hex distribution visualization
    sample_hex_values = [0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 
                         0x90, 0xA0, 0xB0, 0xC0, 0xD0, 0xE0, 0xF0, 0xFF]
    dist_viz = hex_converter.visualize_hex_distribution(sample_hex_values)
    print(f"Hex Distribution Visualization:")
    print(dist_viz)
    
    # Demonstrate composition visualization
    sample_composition = {
        'composition_id': 'SAMPLE_COMP_001',
        'efficiency_score': 0.75,
        'risk_level': 'medium',
        'hex_values': [0x10, 0x20, 0x30, 0x40, 0x50],
        'resources': {
            'compute': {'hex_value': 0x10, 'allocation': 10.5, 'priority': 1},
            'memory': {'hex_value': 0x20, 'allocation': 20.0, 'priority': 2},
            'gpu': {'hex_value': 0x40, 'allocation': 40.0, 'priority': 4}
        }
    }
    
    comp_viz = hex_converter.hex_composition_to_ascii(sample_composition)
    print(f"Composition Visualization:")
    print(comp_viz)
    
    # Create a simple test image programmatically
    try:
        # Create a simple test image
        width, height = 20, 10
        image = Image.new('RGB', (width, height), 'white')
        pixels = image.load()
        
        # Draw a simple pattern
        for x in range(width):
            for y in range(height):
                if (x // 4) % 2 == (y // 2) % 2:  # Checkerboard pattern
                    pixels[x, y] = (128, 128, 128)  # Gray
        
        # Render the test image as ASCII
        renderer = ASCIIImageRenderer(ASCIIConfig(width=40, height=20))
        ascii_image = renderer.render_image(image)
        print(f"\nGenerated Image as ASCII Art:")
        print(ascii_image)
        
    except ImportError:
        print("\n[INFO] Pillow not available, skipping image rendering demo")
    
    print("\n" + "=" * 80)
    print("ASCII IMAGE RENDERING DEMONSTRATION COMPLETE")
    print("System provides:")
    print("- Hexadecimal to ASCII art conversion")
    print("- Distribution visualization")
    print("- Composition visualization")
    print("- Image to ASCII conversion")
    print("=" * 80)


def create_sample_image():
    """Create a sample image for testing."""
    # This function would create a sample image
    # For now, we'll just demonstrate the capability
    pass


if __name__ == "__main__":
    demo_ascii_image_rendering()