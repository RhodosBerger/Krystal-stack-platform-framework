"""
Setup script for GAMESA TPU Optimization Framework

This script ensures all TPU-specific components are properly integrated
into the GAMESA ecosystem and can be installed as part of the project.
"""

import os
import sys
from setuptools import setup, find_packages

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

setup(
    name="gamesa-tpu-optimization",
    version="1.0.0",
    description="GAMESA TPU Optimization Framework - Advanced TPU performance optimization with economic trading and cognitive assistance",
    author="GAMESA Development Team",
    author_email="gamesa@optimization.com",
    packages=find_packages(where="src", include=["tpu_*", "gamesa_tpu_integration"]),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies would go here
        # These would match the existing project requirements
    ],
    extras_require={
        "tpu": ["intel-extension-for-python", "openvino-dev"],  # TPU-specific extras
    },
    entry_points={
        "console_scripts": [
            "gamesa-tpu-demo=src.python.demo_tpu_optimization:main",
            "gamesa-tpu-test=src.python.test_tpu_components:run_all_tests",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Hardware :: Symmetric Multi-processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="tpu optimization, machine learning, performance, economics, cognitive systems",
    project_urls={
        "Source": "https://github.com/gamesa-dev/GAMESA-TPU-Optimization",
        "Tracker": "https://github.com/gamesa-dev/GAMESA-TPU-Optimization/issues",
    },
)