import time
import torch
import numpy as np

# Assuming your custom Tensor class is imported as MyTensor
from autodiff import Tensor  # Update this import as needed
import autodiff as ad

# Matrix size
N = 128

# Generate random data
a_np = np.random.randn(N, N).astype(np.float32)
b_np = np.random.randn(N, N).astype(np.float32)

# PyTorch tensors
a_torch = torch.from_numpy(a_np)
b_torch = torch.from_numpy(b_np)

# Your Tensor class
a_my = Tensor((N, N), zeros_like=False)
b_my = Tensor((N, N), zeros_like=False)

# Warmup
_ = a_torch @ b_torch
_ = ad.matmul(a_my, b_my)

# Time PyTorch matmul
start = time.time()
for _ in range(10):
    c_torch = a_torch @ b_torch
torch_time = time.time() - start

# Time your Tensor matmul
start = time.time()
for _ in range(10):
    c_my = ad.matmul(a_my, b_my)
my_time = time.time() - start

print(f"PyTorch matmul time: {torch_time:.4f} seconds")
print(f"My Tensor matmul time: {my_time:.4f} seconds")
print(f"Speedup (PyTorch / MyTensor): {torch_time / my_time:.2f}x")