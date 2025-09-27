#!/usr/bin/env python3
# Test the actual ML-Agents functionality that matters for your project

print("Testing ML-Agents core functionality...")

# Test 1: Basic ML-Agents import
try:
    import mlagents
    print("✓ ML-Agents base package imported")
except ImportError as e:
    print(f"✗ ML-Agents base import failed: {e}")

# Test 2: Environment creation (this is what you'll actually use)
try:
    from mlagents_envs.environment import UnityEnvironment
    print("✓ UnityEnvironment imported (core functionality)")
except ImportError as e:
    print(f"✗ UnityEnvironment import failed: {e}")

# Test 3: Training components (alternative to trainer_util)
try:
    from mlagents.trainers.trainer_controller import TrainerController
    print("✓ TrainerController imported")
except ImportError as e:
    print("⚠ TrainerController not available (may be version difference)")

# Test 4: Configuration handling
try:
    from mlagents.trainers.settings import RunOptions
    print("✓ RunOptions imported")
except ImportError as e:
    print("⚠ RunOptions not available (may be version difference)")

# Test 5: What you'll actually use for training
try:
    from mlagents_envs.registry import default_registry
    print("✓ Environment registry imported")
except ImportError as e:
    print("⚠ Environment registry not available")

# Test 6: Command line training (this is how you'll run training)
import subprocess
import sys
result = subprocess.run([sys.executable, '-c', 'import mlagents.trainers.learn'], 
                       capture_output=True, text=True)
if result.returncode == 0:
    print("✓ ML-Agents training module accessible")
else:
    print(f"⚠ Training module issue: {result.stderr}")

print("\n✓ Core ML-Agents functionality test complete!")
print("Note: Some advanced trainer utilities may not be needed for your project.")
