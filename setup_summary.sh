#!/bin/bash

# OpenPI Setup Summary
# This script summarizes the setup steps performed to get openpi working
# with the virtual environment on a system with manylinux_2_28_x86_64 platform

echo "=== OpenPI Setup Summary ==="
echo "Platform: manylinux_2_28_x86_64 (RHEL/CentOS 8)"
echo "Python: 3.11.11"
echo "Package Manager: uv"
echo ""

# Step 1: Navigate to project directory
echo "1. Navigate to project directory:"
echo "   cd /home/ftesshu/openpi"
echo ""

# Step 2: Install core openpi package without dependencies
echo "2. Install core openpi package without dependencies:"
echo "   uv pip install -e . --no-deps"
echo ""

# Step 3: Install main dependencies manually (avoiding rerun-sdk platform issue)
echo "3. Install main dependencies manually:"
echo "   uv pip install \\"
echo "     \"augmax>=0.3.4\" \"dm-tree>=0.1.8\" \"einops>=0.8.0\" \\"
echo "     \"equinox>=0.11.8\" \"flatbuffers>=24.3.25\" \"flax==0.10.2\" \\"
echo "     \"fsspec[gcs]>=2024.6.0\" \"imageio>=2.36.1\" \"jax[cuda12]==0.5.3\" \\"
echo "     \"jaxtyping==0.2.36\" \"ml_collections==1.0.0\" \"numpy>=1.22.4,<2.0.0\" \\"
echo "     \"numpydantic>=1.6.6\" \"opencv-python>=4.10.0.84\" \"orbax-checkpoint==0.11.13\" \\"
echo "     \"pillow>=11.0.0\" \"sentencepiece>=0.2.0\" \"torch>=2.7.0\" \\"
echo "     \"tqdm-loggable>=0.2\" \"typing-extensions>=4.12.2\" \"tyro>=0.9.5\" \\"
echo "     \"wandb>=0.19.1\" \"filelock>=3.16.1\" \"beartype==0.19.0\" \\"
echo "     \"treescope>=0.1.7\" \"transformers==4.48.1\" \"rich>=14.0.0\" \"polars>=1.30.0\""
echo ""

# Step 4: Install workspace package
echo "4. Install workspace package:"
echo "   uv pip install -e packages/openpi-client/"
echo ""

# Step 5: Install development dependencies
echo "5. Install development dependencies:"
echo "   uv pip install \\"
echo "     \"pytest>=8.3.4\" \"ruff>=0.8.6\" \"pre-commit>=4.0.1\" \\"
echo "     \"ipykernel>=6.29.5\" \"ipywidgets>=8.1.5\" \"matplotlib>=3.10.0\" \\"
echo "     \"pynvml>=12.0.0\""
echo ""

# Step 6: Install gym-aloha
echo "6. Install gym-aloha:"
echo "   uv pip install \"gym-aloha>=0.1.1\""
echo ""

# Step 7: Install lerobot without dependencies
echo "7. Install lerobot without dependencies:"
echo "   uv pip install git+https://github.com/huggingface/lerobot.git@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5 --no-deps"
echo ""

# Step 8: Install rerun-sdk with pre-release
echo "8. Install rerun-sdk with pre-release:"
echo "   uv pip install --prerelease=allow rerun-sdk"
echo ""

# Step 9: Fix numpy version conflict
echo "9. Fix numpy version conflict (rerun-sdk upgraded numpy to 2.x, but openpi requires <2.0.0):"
echo "   uv pip install \"numpy>=1.22.4,<2.0.0\" --force-reinstall"
echo ""

# Verification commands
echo "=== Verification Commands ==="
echo "Test openpi import:"
echo "   python -c \"import openpi; print('openpi imported successfully')\""
echo ""
echo "Test training config:"
echo "   python -c \"from openpi.training import config; print('Training config imported successfully')\""
echo ""
echo "Test policy config:"
echo "   python -c \"from openpi.policies import policy_config; print('Policy config imported successfully')\""
echo ""
echo "Test rerun-sdk:"
echo "   python -c \"import rerun; print('rerun-sdk imported successfully')\""
echo ""
echo "Check versions:"
echo "   python -c \"import rerun; import numpy; print('rerun-sdk:', rerun.__version__, 'numpy:', numpy.__version__)\""
echo ""

echo "=== Key Points ==="
echo "• Used project's uv virtual environment (not global installation)"
echo "• Avoided platform compatibility issues with rerun-sdk by installing in correct context"
echo "• Installed dependencies manually to work around lerobot's rerun-sdk requirement"
echo "• Fixed numpy version conflict (rerun-sdk upgraded to numpy 2.x, but openpi requires <2.0.0)"
echo "• All core openpi functionality is now available"
echo "• Development tools and testing frameworks are installed"
echo ""
echo "=== Final Package Versions ==="
echo "• rerun-sdk: 0.24.0-alpha.4"
echo "• numpy: 1.26.4 (compatible with openpi requirements)"
echo "• lerobot: 0.1.0"
echo "• openpi: 0.1.0"
echo ""
echo "=== Next Steps ==="
echo "• You can now run openpi models and policies"
echo "• Fine-tune models on your own data"
echo "• Run inference with pre-trained checkpoints"
echo "• Use development tools for testing and development" 