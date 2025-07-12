from autodiff import Variable
import torch
import time

"""
Gradient descent implemented for root finding using automatic differentiation.
"""


def func(x1, x2):
    """Himmelblau's function has four identical minima at:
        (3.0, 2.0)
        (-2.805118, 3.131312)
        (-3.779310, -3.283186)
        (3.584428, -1.848126)
    """
    return ((x1 ** 2 + x2 - 11) ** 2) + ((x1 + x2 ** 2 - 7) ** 2)


args = {
    'x1': -5.,
    'x2': 2.,
    'alpha': 0.01,
    'iterations': int(1e5),
    'eps': 1e-31
}


def gd(x1, x2, alpha, iterations, eps=1e-24):
    x1_var = Variable(x1)
    x2_var = Variable(x2)

    for i in range(iterations):
        z = func(x1_var, x2_var)

        if abs(z.data) < eps:
            return x1_var.data, x2_var.data, i

        z.backward()

        x1_var.data -= alpha * x1_var.grad
        x2_var.data -= alpha * x2_var.grad

        x1_var.zero_grad()
        x2_var.zero_grad()

    return x1_var.data, x2_var.data, i


def gd_torch(x1, x2, alpha, iterations, eps=1e-24):
    x1_var = torch.tensor(x1, dtype=float, requires_grad=True)
    x2_var = torch.tensor(x2, dtype=float, requires_grad=True)

    for i in range(iterations):
        z = func(x1_var, x2_var)

        if abs(z) < eps:
            return x1_var.data, x2_var.data, i

        z.backward()

        x1_var.data -= alpha * x1_var.grad
        x2_var.data -= alpha * x2_var.grad

        x1_var.grad = None
        x2_var.grad = None

    return x1_var.data, x2_var.data, i


start = time.time()
x1_opt, x2_opt, i = gd_torch(**args)
torch_time = time.time() - start

start = time.time()
x1_opt, x2_opt, i = gd(**args)
autodiff_time = time.time() - start

print(f"AUTODIFF: Optimized values: x1 = {x1_opt}, x2 = {x2_opt}, \
      z = {func(x1_opt, x2_opt)}, after {i} iterations in {autodiff_time} seconds.")

print(f"TORCH: Optimized values: x1 = {x1_opt}, x2 = {x2_opt}, \
      z = {func(x1_opt, x2_opt)}, after {i} iterations in {torch_time} seconds.")
