#!/usr/bin/env python3
import sys
print("Python version:", sys.version)

try:
    import mlagents
    print("✓ ML-Agents imported successfully")
except ImportError as e:
    print("✗ ML-Agents import failed:", e)

try:
    import torch
    print("✓ PyTorch imported successfully")
except ImportError as e:
    print("✗ PyTorch import failed:", e)

try:
    import numpy
    print("✓ NumPy imported successfully")
except ImportError as e:
    print("✗ NumPy import failed:", e)

try:
    import yaml
    print("✓ YAML imported successfully")
except ImportError as e:
    print("✗ YAML import failed:", e)

try:
    import psutil
    print("✓ psutil imported successfully")
except ImportError as e:
    print("✗ psutil import failed:", e)

print("\n✓ Setup verification complete!")
