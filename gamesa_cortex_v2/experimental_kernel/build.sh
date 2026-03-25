#!/bin/bash
# Build script for Experimental Kernel
# Author: Dušan Kopecký <dusan.kopecky0101@gmail.com>

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "🔨 Building Experimental Kernel for Krystal Stack"
echo "============================================================"

# Check for Rust
if ! command -v cargo &> /dev/null; then
    echo "❌ Error: cargo (Rust) is not installed"
    echo "   Install from: https://rustup.rs/"
    exit 1
fi

echo "✓ Rust found: $(rustc --version)"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"

# Build Rust library
echo ""
echo "📦 Building Rust library..."
cargo build --release

echo "✓ Build complete!"

# Check for maturin
if ! command -v maturin &> /dev/null; then
    echo ""
    echo "⚠️  maturin not found. Installing..."
    pip3 install maturin
fi

# Build Python bindings
echo ""
echo "🐍 Building Python bindings..."
maturin develop --release

echo "✓ Python bindings installed!"

# Run tests
echo ""
echo "🧪 Running tests..."
cargo test

echo ""
echo "============================================================"
echo "✅ BUILD COMPLETE"
echo "============================================================"
echo ""
echo "The Experimental Kernel is ready to use!"
echo ""
echo "Quick start:"
echo "  python3 demo.py"
echo ""
echo "Documentation:"
echo "  cat README.md"
echo ""
echo "============================================================"
