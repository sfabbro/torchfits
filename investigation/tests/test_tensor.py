import torchfits.cpp as cpp
import torch

print("Testing tensor return...")
t = cpp.test_tensor_return()
print(f"Success! Type: {type(t)}, Value: {t}")
