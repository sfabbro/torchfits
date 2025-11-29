import sys
import traceback

try:
    import torchfits.cpp as cpp
    print("Module imported successfully")
    print(f"test_tensor_return function: {cpp.test_tensor_return}")
    print("Attempting to call test_tensor_return...")
    result = cpp.test_tensor_return()
    print(f"Success! Got: {result}")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
    print("\n---\nDetailed exception info:")
    print(f"Type: {type(e)}")
    print(f"Args: {e.args}")
