#!/usr/bin/env python
"""Quick test script to check which examples work"""
import subprocess
import os
import sys

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

examples = [
    "example_ml_pipeline.py",
    "example_comprehensive_ml_pipeline.py",
    "example_mnist.py",
    "example_sdss_classification.py",
    "example_phase2_features.py",
]

def run_test():
    print(f"Running tests from: {os.getcwd()}")
    
    # Check if we should adjust paths based on where we are running from
    # If running from root (where src/ is), examples are in examples/
    # If running from examples/, examples are in .
    
    base_dir = "."
    if os.path.exists("examples") and os.path.isdir("examples"):
        base_dir = "examples"
    
    success = True
    
    for example in examples:
        print(f"\n{'='*60}")
        print(f"Testing: {example}")
        print('='*60)
        
        # Construct path to example file
        example_path = os.path.join(base_dir, example)
        if not os.path.exists(example_path) and os.path.exists(os.path.join(SCRIPT_DIR, example)):
             example_path = os.path.join(SCRIPT_DIR, example)
             
        if not os.path.exists(example_path):
            print(f"❌ {example} - SKIPPED (File not found at {example_path})")
            success = False
            continue

        result = subprocess.run(
            ["pixi", "run", "python", example_path],
            cwd=".", # Run in current directory
            capture_output=True,
            text=True,
            timeout=120 # Increased timeout for slow examples
        )
        
        if result.returncode == 0:
            print(f"✅ {example} - PASSED")
        else:
            print(f"❌ {example} - FAILED")
            print("Error output:")
            print(result.stderr[:1000]) # Show more output
            success = False

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(run_test())
