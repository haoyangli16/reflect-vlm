
import triton
import triton.runtime
try:
    from triton.runtime import driver
    print("driver imported successfully")
except ImportError as e:
    print(f"ImportError: {e}")
    print(f"triton version: {triton.__version__}")
    print(f"triton path: {triton.__file__}")
    print(f"triton.runtime path: {triton.runtime.__file__}")

