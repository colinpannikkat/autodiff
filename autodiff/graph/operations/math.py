from autodiff.graph.operations.base import BinOp, UnaryOp
from autodiff.graph.var import Variable, Node
import math


class Neg(UnaryOp):
    """Compute negation of a node."""

    @staticmethod
    def _forward(x: Node) -> float:
        return -x.data

    @staticmethod
    def _backward(inputs: list[Variable]) -> list[float]:
        """The gradient of negation is -1, unless constant.
        z = -x
        dz/dx = -1 (unless constant)
        """

        return [-1 if x.requires_grad else 0 for x in inputs]

    @staticmethod
    def op(x):
        return Neg.forward(x)


class Add(BinOp):
    """Compute addition between two nodes."""

    @staticmethod
    def _forward(x: Node, y: Node) -> float:
        return x.data + y.data

    @staticmethod
    def _backward(inputs: list[Variable]) -> list[float]:
        """The gradient of any node in an addition operation is 1.
        If a node is a constant, its gradient should be zero. Both nodes
        that do not require gradients and Constants are identified by the
        requires_grad attribute.
        z = x + y
        dz/dx = 1 (unless constant)
        dz/dy = 1 (unless constant)
        """

        return [1 if x.requires_grad else 0 for x in inputs]

    @staticmethod
    def op(x, y):
        return Add.forward(x, y)


class Mul(BinOp):
    """Computes multiplication between two nodes."""

    @staticmethod
    def _forward(x: Node, y: Node) -> float:
        return x.data * y.data

    @staticmethod
    def _backward(inputs: list[Variable]) -> list[float]:
        """Compute derivative for the product of two variables:
        z = x * y
        dz/dx = y
        dz/dy = x
        """
        x, y = inputs

        x_grad = y if x.requires_grad else 0
        y_grad = x if y.requires_grad else 0

        return [x_grad, y_grad]

    @staticmethod
    def op(x, y):
        return Mul.forward(x, y)


class Div(BinOp):
    """Computes division between two nodes."""

    @staticmethod
    def _forward(x: Node, y: Node) -> float:
        return x.data / y.data

    @staticmethod
    def _backward(inputs: list[Variable]) -> list[float]:
        """Compute derivative for the division of two variables:
        z = x / y
        dz/dx = 1/y
        dz/dy = -(x^2)/y
        """
        x, y = inputs

        x_grad = 1/y if x.requires_grad else 0
        y_grad = -x/(y*y) if y.requires_grad else 0

        return [x_grad, y_grad]

    @staticmethod
    def op(x, y):
        return Div.forward(x, y)


class Pow(BinOp):
    """Computes power operation between two nodes."""

    @staticmethod
    def _forward(x: Node, y: Node) -> float:
        return x.data ** y.data

    @staticmethod
    def _backward(inputs: list[Variable]) -> list[float]:
        """Compute derivative for the power operation of two variables:
        z = x^y
        dz/dx = y * x^(y-1)
        dz/dy = x^y * log(x)
        """
        x, y = inputs

        x_grad = y * (x ** (y - 1)) if x.requires_grad else 0
        y_grad = (x ** y) * x.log() if y.requires_grad else 0

        return [x_grad, y_grad]

    @staticmethod
    def op(x, y):
        return Pow.forward(x, y)


class Log(UnaryOp):
    """Computes natural logarithm of a node."""

    @staticmethod
    def _forward(x: Node) -> float:
        return math.log(x.data)

    @staticmethod
    def _backward(inputs: list[Variable]) -> list[float]:
        """The gradient of log(x) is 1/x, unless constant.
        z = log(x)
        dz/dx = 1/x (unless constant)
        """

        return [1 / x if x.requires_grad else 0 for x in inputs]

    @staticmethod
    def op(x):
        return Log.forward(x)


class Exp(UnaryOp):
    """Computes exponential function of a node."""

    @staticmethod
    def _forward(x: Node) -> float:
        return math.exp(x.data)

    @staticmethod
    def _backward(inputs: list[Variable]) -> list[float]:
        """The gradient of exp(x) is exp(x), unless constant.
        z = exp(x)
        dz/dx = exp(x) (unless constant)
        """

        return [Exp.forward(x).data if x.requires_grad else 0 for x in inputs]

    @staticmethod
    def op(x):
        return Exp.forward(x)


# Manual overloading of Variable operations

# Addition
Variable.__add__ = Add.op
Variable.__radd__ = Add.op

# Subtraction (derivative is the same with the negation added)
Variable.__sub__ = lambda self, other: Add.op(self, -other)
Variable.__rsub__ = lambda self, other: Add.op(other, -self)

# Multiplication
Variable.__mul__ = Mul.op
Variable.__rmul__ = Mul.op

# Division
Variable.__truediv__ = Div.op
Variable.__rtruediv__ = lambda self, other: Div.op(other, self)
Variable.__itruediv__ = Div.op  # In-place division (x /= y)
Variable.__div__ = Div.op       # Python 2 division (x / y)
Variable.__idiv__ = Div.op      # Python 2 in-place division (x /= y)

# Power
Variable.__pow__ = Pow.op
Variable.__rpow__ = lambda self, other: Pow.op(other, self)

# Negation
Variable.__neg__ = Neg.op

# Log
Variable.log = Log.op

# Exp
Variable.exp = Exp.op
