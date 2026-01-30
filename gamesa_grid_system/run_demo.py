#!/usr/bin/env python3
"""
Run the GAMESA Grid System Demo
"""
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main import demo_gamespace_system

if __name__ == "__main__":
    demo_gamespace_system()