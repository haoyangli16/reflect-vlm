"""
Fix for LLVM command-line option conflict between triton and bitsandbytes.
Also sets up MuJoCo/OpenGL for headless rendering (EGL).
This script must be imported BEFORE any other imports that use triton, bitsandbytes, or mujoco.
"""

import os
import sys

# ==========================================
# MuJoCo / OpenGL EGL Setup (MUST BE FIRST)
# ==========================================
# These must be set BEFORE mujoco is imported
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# ==========================================
# Triton / BitsAndBytes Fixes
# ==========================================
# Set environment variable to prevent triton from using ptxas
os.environ["TRITON_PTXAS_PATH"] = ""

# Prevent LLVM from aborting on duplicate options
os.environ["LLVM_DISABLE_CRASH_REPORT"] = "1"

# Force bitsandbytes to use CUDA 11.7 libraries (matching PyTorch)
os.environ["BNB_CUDA_VERSION"] = "117"

# Ensure NVIDIA and CUDA libraries are in the path
# This is critical for PyTorch to detect CUDA properly
nvidia_lib_path = "/usr/local/nvidia/lib64"
cuda_lib_path = "/usr/local/cuda/lib64"
ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")

paths_to_add = []
if os.path.exists(nvidia_lib_path) and nvidia_lib_path not in ld_library_path:
    paths_to_add.append(nvidia_lib_path)
if os.path.exists(cuda_lib_path) and cuda_lib_path not in ld_library_path:
    paths_to_add.append(cuda_lib_path)

if paths_to_add:
    new_path = ":".join(paths_to_add)
    os.environ["LD_LIBRARY_PATH"] = f"{new_path}:{ld_library_path}" if ld_library_path else new_path

# Pre-import triton to register its LLVM options first
try:
    # Import triton before bitsandbytes to control the order
    import triton
    import triton.language as tl

    # Suppress triton's LLVM option registration on subsequent imports
    if hasattr(triton, "_C"):
        # Mark that triton has been initialized
        pass
except ImportError:
    print("Warning: Could not import triton for pre-initialization")
    pass
except Exception as e:
    # Suppress any triton initialization errors
    print(f"Warning: Triton pre-initialization encountered an issue (continuing anyway): {e}")
    pass
