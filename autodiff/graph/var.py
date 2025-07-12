from autodiff.graph.node import Node
from autodiff.config import is_higher_order_gradients_enabled
import math


class Variable(Node):
    """This class represents a floating point variable in a computational graph.

    This class can take in an initial value if instantitate directly, or can also
    be the result of any operation in a computational graph, which would then be
    used to compute the gradient during the backward pass.
    """

    def __init__(
        self,
        value: float,
        requires_grad: bool = True,
        op=None
    ):
        super().__init__(requires_grad)

        self.data = float(value)
        self.op = op
        self.higher_order = is_higher_order_gradients_enabled()

    def __repr__(self):
        return f"Variable({self.data}, grad={self.grad})"

    def forward(self, *args, **kwargs):
        return self.data

    def backward(
        self,
        grad=None,
        retain_graph: bool = False,
        *args,
        **kwargs
    ):

        if grad is None:
            grad = Variable(1.) if self.higher_order else 1.

        if self.grad is not None:
            self.grad += grad
        else:
            self.grad = grad

        # Compute backward pass according to the operation and parent nodes using
        # the chain rule
        if self.op:
            self.op.backward(self.grad, retain_graph=retain_graph)

        # Delete computation graph references if retain is not set to true
        if not retain_graph:
            del self.op
            self.op = None

    def zero_grad(self):
        del self.grad
        self.grad = None


class Constant(Node, float):
    """Represents a constant value in the computational graph. Gradient is always zero."""

    def __init__(self, value: float):
        super().__init__(requires_grad=False)
        self.data = float(value)
        self.grad = 0.0

    def __repr__(self):
        return f"Constant({self.data})"

    def forward(self, *args, **kwargs):
        return self.data

    def backward(self, grad: float = None, *args, **kwargs):
        # Gradient for constants is always zero, do nothing
        pass

    def zero_grad(self):
        self.grad = 0.0

    def log(self):
        return math.log(self)


class Float(float):

    def __init__(self, value: float, requires_grad=True):
        super().__init__()
        self.data = float(value)
        self.grad = 0.0
        self.requires_grad = requires_grad

    def __new__(cls, value, requires_grad=False):
        obj = float.__new__(cls, value)
        obj.requires_grad = requires_grad
        return obj

    def __repr__(self):
        return f"Float({self.data})"

    def log(self):
        return math.log(self)
