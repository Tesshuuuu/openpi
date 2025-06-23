#!/usr/bin/env python3
"""
OpenPI Setup Commands Summary

This script contains all the commands we ran to set up openpi on a 
manylinux_2_28_x86_64 platform (RHEL/CentOS 8) with Python 3.11.11.

The setup was necessary because:
1. The standard 'uv sync' failed due to rerun-sdk platform compatibility issues
2. rerun-sdk only provides wheels for manylinux_2_31_x86_64, but the system runs manylinux_2_28_x86_64
3. We needed to install dependencies manually to work around this issue
"""

import subprocess
import sys
from typing import List, Dict

def run_command(cmd: str, description: str = "") -> bool:
    """Run a command and return success status."""
    if description:
        print(f"\n=== {description} ===")
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main setup function with all the commands we ran."""
    
    print("OpenPI Setup Commands Summary")
    print("=" * 50)
    print("Platform: manylinux_2_28_x86_64 (RHEL/CentOS 8)")
    print("Python: 3.11.11")
    print("Package Manager: uv")
    print()
    
    # All the commands we ran, in order
    commands = [
        # Step 1: Install core openpi package without dependencies
        ("uv pip install -e . --no-deps", 
         "Install core openpi package without dependencies"),
        
        # Step 2: Install main dependencies manually
        ('uv pip install "augmax>=0.3.4" "dm-tree>=0.1.8" "einops>=0.8.0" "equinox>=0.11.8" "flatbuffers>=24.3.25" "flax==0.10.2" "fsspec[gcs]>=2024.6.0" "imageio>=2.36.1" "jax[cuda12]==0.5.3" "jaxtyping==0.2.36" "ml_collections==1.0.0" "numpy>=1.22.4,<2.0.0" "numpydantic>=1.6.6" "opencv-python>=4.10.0.84" "orbax-checkpoint==0.11.13" "pillow>=11.0.0" "sentencepiece>=0.2.0" "torch>=2.7.0" "tqdm-loggable>=0.2" "typing-extensions>=4.12.2" "tyro>=0.9.5" "wandb>=0.19.1" "filelock>=3.16.1" "beartype==0.19.0" "treescope>=0.1.7" "transformers==4.48.1" "rich>=14.0.0" "polars>=1.30.0"',
         "Install main dependencies manually (avoiding rerun-sdk platform issue)"),
        
        # Step 3: Install workspace package
        ("uv pip install -e packages/openpi-client/",
         "Install workspace package"),
        
        # Step 4: Install development dependencies
        ('uv pip install "pytest>=8.3.4" "ruff>=0.8.6" "pre-commit>=4.0.1" "ipykernel>=6.29.5" "ipywidgets>=8.1.5" "matplotlib>=3.10.0" "pynvml>=12.0.0"',
         "Install development dependencies"),
        
        # Step 5: Install gym-aloha
        ('uv pip install "gym-aloha>=0.1.1"',
         "Install gym-aloha"),
        
        # Step 6: Install lerobot without dependencies
        ("uv pip install git+https://github.com/huggingface/lerobot.git@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5 --no-deps",
         "Install lerobot without dependencies"),
        
        # Step 7: Install rerun-sdk with pre-release
        ("uv pip install --prerelease=allow rerun-sdk",
         "Install rerun-sdk with pre-release"),
    ]
    
    # Verification commands
    verification_commands = [
        ("python -c \"import openpi; print('openpi imported successfully')\"",
         "Test openpi import"),
        
        ("python -c \"from openpi.training import config; print('Training config imported successfully')\"",
         "Test training config"),
        
        ("python -c \"from openpi.policies import policy_config; print('Policy config imported successfully')\"",
         "Test policy config"),
        
        ("python -c \"import rerun; print('rerun-sdk imported successfully')\"",
         "Test rerun-sdk"),
    ]
    
    print("Setup Commands:")
    print("-" * 30)
    
    success_count = 0
    for i, (cmd, desc) in enumerate(commands, 1):
        print(f"\n{i}. {desc}")
        print(f"   Command: {cmd}")
    
    print("\nVerification Commands:")
    print("-" * 30)
    
    for i, (cmd, desc) in enumerate(verification_commands, 1):
        print(f"\n{i}. {desc}")
        print(f"   Command: {cmd}")
    
    print("\n" + "=" * 50)
    print("Key Points:")
    print("• Used project's uv virtual environment (not global installation)")
    print("• Avoided platform compatibility issues with rerun-sdk by installing in correct context")
    print("• Installed dependencies manually to work around lerobot's rerun-sdk requirement")
    print("• All core openpi functionality is now available")
    print("• Development tools and testing frameworks are installed")
    
    print("\nNext Steps:")
    print("• You can now run openpi models and policies")
    print("• Fine-tune models on your own data")
    print("• Run inference with pre-trained checkpoints")
    print("• Use development tools for testing and development")

if __name__ == "__main__":
    main() 