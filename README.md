# Autodiff

Autodiff is a reverse-mode auto differentiation engine with dynamic graph construction for backpropagation. It is currently twice as fast as PyTorch for basic backpropagation and gradient descent.

## Installation

You can install the package by cloning the repository and entering the following:

```bash
pip install .
```

Then, instantiate a variable and do any computation

```python3
from autodiff import Variable

x = Variable(2)

y = x ** 2 + x + 2  # 8
```

You can then call backward on a variable to compute the partial derivative w.r.t that variable and every other variable included in the computational graph.

```python3
y.backward()

print(x.grad)  # 5
```
