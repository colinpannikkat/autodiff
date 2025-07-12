from abc import ABC, abstractmethod
from autodiff.graph.var import Variable, Constant


class Op(ABC):
    """This class represents an operation variable in a computational graph.

    This class is used as a an abstract base class for any operations that must
    be derived. Any result of the operation is stored in the self.data attribute.
    """

    def __init__(
        self,
        *inputs
    ):
        super().__init__()
        self.inputs = [*inputs]
        self.desc = "Op"
        self.grads = None
        self.higher = False

    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    @staticmethod
    @abstractmethod
    def _forward(*args, **kwargs) -> float:
        """Overideable forward function to return a float. Used to create a new
        variable in a graph."""
        pass

    @staticmethod
    @abstractmethod
    def _backward(*args, **kwargs) -> list[Variable]:
        """Overideable backward function to gradients. Used to pass gradients to
        parents of the operation."""
        pass

    @staticmethod
    def _require_grad(*args):
        return any(getattr(arg, "requires_grad", False) for arg in args)

    @staticmethod
    def _coerce_primitive_to_constant(*args):
        return [arg if (isinstance(arg, Variable) or isinstance(arg, Constant)) else Constant(arg) for arg in args]

    @classmethod
    def forward(cls, *args, **kwargs) -> Variable:
        """Computes a forward pass operation and returns a new variable node in a graph."""
        args = cls._coerce_primitive_to_constant(*args)
        return Variable(
            cls._forward(*args, **kwargs),
            parents=[*args],
            op=cls(*args, **kwargs),
            requires_grad=cls._require_grad(*args)
        )

    def backward(self, grad: Variable, *args, **kwargs) -> None:
        """Computes the gradients of the arguments of the operation and passes them
        to the inputs of the operation (parents of the result)."""

        self.higher = kwargs.get("allow_higher_order", False)  # higher order gradients

        if self.grads is None:
            self.grads = self._backward()
        for parent, paren_grad in zip(self.inputs, self.grads):
            parent.backward(paren_grad * grad, allow_higher_order=self.higher)


class UnaryOp(Op):
    """This class derives from Op to take a singular operation argument."""
    def __init__(
        self,
        x
    ):
        super().__init__(x)
        self.x = x

    @classmethod
    @abstractmethod
    def _forward(cls, x, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def op(x):
        """This is used when overloading an operation in some Variable derived class."""
        pass


class BinOp(Op):
    """This class derives from Op to take binary operation arguments."""
    def __init__(
        self,
        x,
        y
    ):
        super().__init__(x, y)

    @classmethod
    @abstractmethod
    def _forward(cls, x, y, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def op(x, y):
        """This is used when overloading an operation in some Variable derived class."""
        pass
