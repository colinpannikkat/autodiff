import time
import torch
from autodiff import Variable

"""
Speed comparison between autodiff and pytorch for forward and backward pass.
"""


def func(x1, x2):
    """Himmelblau's function"""
    return ((x1 ** 2 + x2 - 11) ** 2) + ((x1 + x2 ** 2 - 7) ** 2)


def autodiff_backward(x):
    x1 = Variable(x)
    x2 = Variable(x)
    y = func(x1, x2)
    y.backward()
    return x1.grad, x2.grad


def pytorch_backward(x):
    x1 = torch.tensor(x, requires_grad=True)
    x2 = torch.tensor(x, requires_grad=True)
    y = func(x1, x2)
    y.backward()
    return x1.grad.item(), x2.grad.item()


def speed_comparison():
    x = 2.0
    iterations = 10000

    start = time.time()
    for _ in range(iterations):
        autodiff_backward(x)
    autodiff_time = time.time() - start

    start = time.time()
    for _ in range(iterations):
        pytorch_backward(x)
    pytorch_time = time.time() - start

    print(f"Autodiff time: {autodiff_time:.6f} seconds")
    print(f"PyTorch time: {pytorch_time:.6f} seconds")


if __name__ == "__main__":
    speed_comparison()