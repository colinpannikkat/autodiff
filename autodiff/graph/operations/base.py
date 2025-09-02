from abc import ABC, abstractmethod
from autodiff.graph.var import Variable, Constant, Float
from autodiff.config import is_higher_order_gradients_enabled


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
        self.higher_order = is_higher_order_gradients_enabled()

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
    def _backward(*args, **kwargs) -> list[float]:
        """Overideable backward function to gradients. Used to pass gradients to
        parents of the operation."""
        pass

    @staticmethod
    def _require_grad(*args):
        return any(getattr(arg, "requires_grad", False) for arg in args)

    @staticmethod
    def _convert_primitive_to_constant(*args):
        return [arg if (isinstance(arg, Variable) or isinstance(arg, Constant)) else Constant(arg) for arg in args]

    @staticmethod
    def _convert_primitive_to_variable(*args):
        return [arg if (isinstance(arg, Variable) or isinstance(arg, Constant)) else Variable(arg) for arg in args]

    @staticmethod
    def _convert_variable_to_float(*args):
        return [Float(arg.data, arg.requires_grad) if isinstance(arg, Variable) else arg for arg in args]

    @classmethod
    def forward(cls, *args, **kwargs) -> Variable:
        """Computes a forward pass operation and returns a new variable node in a graph."""
        args = cls._convert_primitive_to_constant(*args)

        return Variable(
            cls._forward(*args, **kwargs),
            op=cls(*args, **kwargs),
            requires_grad=cls._require_grad(*args)
        )

    def backward(self, grad: Variable | float, *args, **kwargs) -> None:
        """Computes the gradients of the arguments of the operation and passes them
        to the inputs of the operation (parents of the result)."""

        if self.higher_order:
            inputs = self.inputs
        else:
            inputs = self._convert_variable_to_float(*self.inputs)

        if self.grads is None:
            self.grads = self._backward(inputs)

        if self.higher_order:
            self.grads = self._convert_primitive_to_variable(*self.grads)

        for parent, paren_grad in zip(self.inputs, self.grads):
            parent.backward(paren_grad * grad, **kwargs)


class UnaryOp(Op):
    """This class derives from Op to take a singular operation argument."""
    def __init__(
        self,
        x
    ):
        super().__init__(x)

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
    def _backward(*args, **kwargs) -> list[float]:
        """Overridable backward function to gradients. Used to pass gradients to
        parents of the operation."""
        pass

    @staticmethod
    @abstractmethod
    def op(x, y):
        """This is used when overloading an operation in some Variable derived class."""
        pass
