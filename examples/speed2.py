import time
import torch
from autodiff import Variable
from autodiff.func import relu, cross_entropy_loss

"""
Speed comparison between autodiff and pytorch for forward and backward pass.

Simulating 2-layer NN forward pass, with cross_entropy_loss at end.
"""

def autodiff(x, y):
    W1 = Variable([[0.5, -0.2], [0.3, 0.8]])
    b1 = Variable([0.1, -0.1])
    W2 = Variable([[0.7], [-0.5]])
    b2 = Variable([0.2])

    h = relu(x @ W1 + b1)
    y_hat = h @ W2 + b2

    loss = cross_entropy_loss(y_hat, y)

    loss.backward()

def pytorch(x, y):
    W1 = torch.tensor([[0.5, -0.2], [0.3, 0.8]], requires_grad=True)
    b1 = torch.tensor([0.1, -0.1], requires_grad=True)
    W2 = torch.tensor([[0.7], [-0.5]], requires_grad=True)
    b2 = torch.tensor([0.2], requires_grad=True)

    h = torch.relu(x @ W1 + b1)
    y_hat = h @ W2 + b2

    loss = torch.nn.functional.cross_entropy(y_hat, y)

    loss.backward()

def speed_comparison():
    x_torch = torch.tensor([[1.0, 2.0]])
    y_torch = torch.tensor([[1.0]])
    x_autodiff = Variable([[1.0, 2.0]])
    y_autodiff = Variable([[1.0]])
    iterations = 10000

    start = time.time()
    for _ in range(iterations):
        autodiff(x_autodiff, y_autodiff)
    autodiff_time = time.time() - start

    start = time.time()
    for _ in range(iterations):
        pytorch(x_torch, y_torch)
    pytorch_time = time.time() - start

    print(f"Autodiff time: {autodiff_time:.6f} seconds")
    print(f"PyTorch time: {pytorch_time:.6f} seconds")

if __name__ == "__main__":
    speed_comparison()