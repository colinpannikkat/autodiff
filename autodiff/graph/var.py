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

        if grad is None:  # with respect to itself
            grad = Constant(1.) if self.higher_order else 1.

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

    def zero_grad(self, visited: set = None, retain_grad: bool = False):
        """ Deletes the gradient for the corresponding Variable.

        Inputs:
            visited (set): Visited objects, prevents infinite loops when freeing
                objects in computational graph through topological ordering. For
                example, when building gradients the gradient is computed via the
                node that it is a gradient of, so when calling through zero_grad,
                it will revisit the parent node. NOT TO BE USED BY USER.
            retain_grad (bool, default=False): Retain the computational graph
                gradients of all other nodes in the computational graph.
        """
        # Ensure no gradients are preserved in the graph
        if visited is None:
            visited = set([id(self)])
        else:
            visited.add(id(self))

        if self.op and not retain_grad:
            for input in self.op.inputs:
                if id(input) not in visited:
                    input.zero_grad(visited)

        if self.grad is None:
            return

        # Wipe gradient info in if higher_order
        if self.higher_order:
            self.grad.zero_grad(visited)

        del self.grad
        self.grad = None


class Constant(Variable, float):
    """Represents a constant value in the computational graph. Gradient is always zero."""

    def __init__(self, value: float):
        super().__init__(value, requires_grad=False)
        self.data = float(value)
        self.grad = 0.0

    def __repr__(self):
        return f"Constant({self.data})"

    def forward(self, *args, **kwargs):
        return self.data

    def backward(self, *args, **kwargs):
        # Gradient for constants is always zero, do nothing
        self.grad = 0.0

    def zero_grad(self, *args, **kwargs):
        pass

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

    def backward(self, *args, **kwargs):
        # Gradient for constants is always zero, do nothing
        self.grad = 0.0

    def zero_grad(self, *args, **kwargs):
        # Gradient for constants is always zero, do nothing
        self.grad = 0.0

    def __repr__(self):
        return f"Float({self.data})"

    def log(self):
        return math.log(self)
