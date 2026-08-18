# ASCII Image Renderer for GAMESA Framework

This module implements an ASCII image renderer that converts images to ASCII art and integrates with the hexadecimal system and existing framework components.

## Components

### 1. ASCII Image Renderer (`ascii_image_renderer.py`)

The ASCII Image Renderer provides capabilities to convert images and data to ASCII representations:

- **Image to ASCII Conversion**: Converts images to ASCII art with customizable parameters
- **Configurable Output**: Adjustable width, height, character set, and scale factor
- **Multiple Input Sources**: Support for file paths, bytes, and URLs
- **Grayscale Processing**: Automatic conversion to grayscale for ASCII rendering

#### Key Features:
- Configurable ASCII character sets
- Aspect ratio correction for terminal display
- Multiple image input formats
- Integration with hexadecimal data visualization

### 2. Hexadecimal ASCII Converter

The Hexadecimal ASCII Converter provides tools to visualize hexadecimal data:

- **Hex to ASCII Art**: Converts hexadecimal strings to readable ASCII representations
- **Distribution Visualization**: Creates visualizations of hex value distributions
- **Composition Visualization**: Converts composition data to ASCII art
- **Pattern Recognition**: Visualizes patterns in hex data

## Installation

The ASCII renderer is part of the broader GAMESA framework and requires:

1. Core dependencies:
```bash
pip install Pillow numpy
```

2. For full functionality:
```bash
pip install requests  # For URL image loading
```

## Usage

### Basic Image Rendering

```python
from ascii_image_renderer import ASCIIImageRenderer, ASCIIConfig
from PIL import Image

# Create renderer with custom configuration
config = ASCIIConfig(width=80, height=40, charset=" .:-=+*#%@")
renderer = ASCIIImageRenderer(config)

# Render an image
image = Image.open("path/to/image.jpg")
ascii_art = renderer.render_image(image)
print(ascii_art)
```

### Hexadecimal Data Visualization

```python
from ascii_image_renderer import HexadecimalASCIIConverter

# Create converter
converter = HexadecimalASCIIConverter()

# Convert hex string to ASCII art
hex_data = "48656C6C6F20576F726C64"  # "Hello World" in hex
ascii_art = converter.hex_to_ascii_art(hex_data)
print(ascii_art)

# Visualize hex distribution
hex_values = [0x10, 0x20, 0x30, 0x40, 0x50]
dist_viz = converter.visualize_hex_distribution(hex_values)
print(dist_viz)

# Visualize composition data
composition_data = {
    'composition_id': 'SAMPLE_COMP_001',
    'efficiency_score': 0.75,
    'risk_level': 'medium',
    'hex_values': [0x10, 0x20, 0x30, 0x40, 0x50],
    'resources': {
        'compute': {'hex_value': 0x10, 'allocation': 10.5, 'priority': 1}
    }
}
comp_viz = converter.hex_composition_to_ascii(composition_data)
print(comp_viz)
```

### Advanced Configuration

```python
from ascii_image_renderer import ASCIIConfig

# Custom character set for different visual effects
dark_theme_config = ASCIIConfig(
    width=100,
    height=50,
    charset="@%#*+=-:. ",
    invert=True
)

light_theme_config = ASCIIConfig(
    width=100,
    height=50,
    charset=" .:-=+*#%@",
    invert=False
)
```

## Integration with GAMESA Framework

The ASCII renderer integrates seamlessly with the existing GAMESA framework:

- **Hexadecimal System**: Visualizes hex trading data and market states
- **Composition Generator**: Renders composition data in ASCII format
- **Telemetry Visualization**: Converts system telemetry to visual formats
- **Resource Management**: Visualizes resource allocation and usage patterns

### Integration Points:
- Hexadecimal trading system visualization
- Composition data rendering
- Market state visualization
- System telemetry display
- Resource allocation patterns

## Configuration Options

### ASCIIConfig Parameters:
- `width`: Output width in characters (default: 80)
- `height`: Output height in characters (default: 40)
- `charset`: Character set for ASCII rendering (default: " .:-=+*#%@")
- `invert`: Invert brightness mapping (default: False)
- `scale_factor`: Character aspect ratio correction (default: 0.43)

### Character Sets:
- **Standard**: " .:-=+*#%@" (light to dark)
- **Reverse**: "@%#*+=-:. " (dark to light)
- **Simple**: " .o*#" (minimal set)
- **Detailed**: " .-~+*#%@M" (more gradations)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  ASCII IMAGE RENDERER                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ Image Processor │  │ Hex Converter   │  │ Config Mgr  │ │
│  │                 │  │                 │  │             │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
│         │                       │                   │       │
│  ┌─────────────────────────────────────────────────────────┤
│  │              ASCII Conversion Engine                    │ │
│  │        (Resize, Grayscale, ASCII Mapping)             │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              VISUALIZATION COMPONENTS                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Hex Visual  │  │ Image Visual│  │ Composition Visual  │ │
│  │ izer        │  │ izer        │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│         │                │                      │            │
│  ┌─────────────────────────────────────────────────────────┤
│  │              Visualization Engine                       │ │
│  │        (Distribution, Pattern, Data Mapping)          │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Features

### Image Rendering
- High-quality ASCII conversion
- Configurable output dimensions
- Custom character sets
- Aspect ratio correction
- Multiple input formats

### Hexadecimal Visualization
- Hex string to ASCII conversion
- Distribution pattern visualization
- Composition data rendering
- Market state visualization
- Resource allocation patterns

### Framework Integration
- Hexadecimal trading system
- Composition generator
- Telemetry visualization
- Resource management
- Market analysis

## Testing

Run the comprehensive test suite:

```bash
python test_ascii_renderer.py
```

This will test all components and verify the integration between image rendering and hexadecimal visualization.

## Performance

The renderer is optimized for:
- Fast image processing
- Memory-efficient conversions
- Configurable quality settings
- Real-time visualization updates
- Batch processing capabilities

## Security

- Input validation for image files
- Safe URL handling (when using requests)
- Memory bounds checking
- Error handling for invalid formats
- Secure file access patterns