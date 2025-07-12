from collections import defaultdict
from abc import ABC
from typing import List, Iterable
import numpy as np

def _cast_to_variable(*args):
    if len(args) == 1:
        if type(args[0]) is not Variable:
            return Variable(args[0])
        else:
            return args[0]
    return tuple(Variable(arg) if type(arg) is not Variable else arg for arg in args)

class Node(ABC):
    """Node abstract base class for building a computational graph."""

    def __init__(self, parents: List['Node'] = None):
        super().__init__()
        self.parents = parents
        self.op = None
        self.grad = None
        self.data = None

class UnaryOp(ABC):
    
    def __init__(self, x):
        super().__init__()

        self.x = x

    def __repr__(self):
        return "<UnaryOp>"

class BinOp(ABC):
    
    def __init__(self, x: Node, y: Node):
        super().__init__()

        self.x = x
        self.y = y

    def __repr__(self):
        return "<BinOp>"

class Add(BinOp):

    def __repr__(self):
        return "<Add>"

    def forward(x, y):
        x, y = _cast_to_variable(x, y)
        return Variable(x.data+y.data, [x, y], op=Add(x, y))
    
    def backward(self, grad):
        return grad, grad

class Mul(BinOp):

    def __repr__(self):
        return "<Mul>"

    def forward(x, y):
        x, y = _cast_to_variable(x, y)
        return Variable(x.data*y.data, [x, y], op=Mul(x, y))
    
    def backward(self, grad):
        dy = grad * self.x.data
        dx = grad * self.y.data

        return dx, dy
        
class MatMul(BinOp):

    def __repr__(self):
        return "<MatMul>"

    def forward(x, y):
        x, y = _cast_to_variable(x, y)
        return Variable(np.matmul(x.data, y.data), [x, y], op=MatMul(x, y))
    
    def backward(self, grad):
        # grad_output is dL/dZ where Z = X @ Y
        dx = np.matmul(grad, self.y.data.T)
        dy = np.matmul(self.x.data.T, grad)

        return dx, dy
        
class Div(BinOp):

    def __repr__(self):
        return "<Div>"

    def forward(x, y):
        x, y = _cast_to_variable(x, y)
        return Variable(x.data / y.data, [x, y], op=Div(x, y))
    
    def backward(self, grad):
        dx = grad / self.y.data
        dy = -grad * self.x.data / (self.y.data ** 2)
        return dx, dy

class Power(BinOp):

    def __repr__(self):
        return "<Power>"

    def forward(x, y):
        x, y = _cast_to_variable(x, y)
        return Variable(np.power(x.data, y.data), [x, y], op=Power(x, y))
    
    def backward(self, grad_output):
        x = self.x.data
        y = self.y.data
        x_safe = np.where(x <= 0, 1e-10, x)  # avoid log(0) or divide-by-zero
        
        # d/dx: y * x^(y-1)
        dx = grad_output * y * np.power(x, y - 1)
        # d/dy: x^y * log(x)
        dy = grad_output * np.power(x, y) * np.log(x_safe)

        return dx, dy
    
class Variable(Node):
    
    def __init__(self, data: float | List = None, parents: List[Node] = None, op = None):
        super().__init__(parents)

        if isinstance(data, float):
            self.data = np.array([data], dtype=float)
        elif isinstance(data, int):
            self.data = np.array([float(data)], dtype=float)
        elif isinstance(data, list):
            self.data = np.array(data, dtype=float)
        elif isinstance(data, np.ndarray):
            self.data = np.array(data, dtype=float)
        else:
            raise TypeError(f"Variable cannot be instantiated with type {type(data)}.")
            
        self.op = op
        self.shape = self.data.shape

    def backward(self, grad=None):

        if grad is None:
            grad = np.ones_like(self.data, dtype=float)

        if self.grad is not None:
            self.grad += grad
        else:
            self.grad = grad

        if self.parents:
            grads = self.op.backward(self.grad)  # returns tuple (dL/dx, dL/dy, ...)

            if not isinstance(grads, tuple):
                grads = (grads,)

            for parent, parent_grad in zip(self.parents, grads):
                parent.backward(parent_grad)
            
            del self.parents # delete computation graph references
            self.parents = None

    def zero_grad(self):
        self.grad = None

    def random(size: Iterable):
        return Variable(np.random.randn(*size))

    def zeros(size: Iterable):
        return Variable(np.zeros(size))
    
    @property
    def T(self):
        return Variable(self.data.T, parents=[self], op=None)

    def __repr__(self):
        if self.op is not None:
            return f"Variable({self.data}, op={self.op})"
        return f"Variable({self.data})"

    def __getitem__(self, index):
        return Variable(self.data[index])
    
    def __neg__(self):
        self.data = -self.data
        return self

    def __mul__(self, other):
        return Mul.forward(self, other)
    
    def __matmul__(self, other):
        return MatMul.forward(self, other)
    
    def __pow__(self, other):
        return Power.forward(self, other)
    
    def __div__(self, other):
        return Div.forward(self, other)
    
    def __truediv__(self, other):
        return self.__div__(other)
    
    def __add__(self, other):
        return Add.forward(self, other)
    
    def __sub__(self, other):
        return Add.forward(self, -other)
    
    def sum(self, dim=None):
        self.data = np.sum(self.data, axis=dim, dtype=float)
        return self
    
    def log(self, b='exp'):
        if b == 'exp':
            self.data = np.log(self.data)
        else:
            self.data = np.emath.logn(b, self.data)
        return self

    def __lt__(self, other):
        return np.less(self.data, other.data)

    def __le__(self, other):
        return np.less_equal(self.data, other.data)

    def __gt__(self, other):
        return np.greater(self.data, other.data)

    def __ge__(self, other):
        return np.greater_equal(self.data, other.data)

    def __eq__(self, other):
        return np.equal(self.data, other.data)

    def __ne__(self, other):
        return np.not_equal(self.data, other.data)