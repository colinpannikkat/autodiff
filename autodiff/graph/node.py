from abc import ABC, abstractmethod


class Node(ABC):
    """This class is an abstract base class for all computational graph nodes.

    The computational graph is represented via linked nodes that contain pointers
    to their parents. This allows for easy reverse topological orderings. Forward
    graph construction is dynamic and automatic given operational overloading in base
    classes.

    An abstract node therein either has a forward or a backward function which must
    be implemented in any inherited sub-node class.

    Attributes:
        - self.grad: Gradient of the node. Default None.
        - self.data: Any data stored in the node. Default None.
        - self.parents: List of parental nodes in the computation graph. Used
                        when computing the backward pass.
        - self.requires_grad: If requires_grad if False, then the corresponding gradient
                              of the node is not computed.
    """

    def __init__(
        self,
        requires_grad: bool = False
    ):
        super().__init__()

        self.requires_grad = requires_grad
        self.grad = None
        self.data = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> "Node":
        """A forward method computes a corresponding operation given the Node properties.
        Since graphs are constructed dynamically there is no explicit forward call needed
        to build the computational graph.
        """
        pass

    @abstractmethod
    def backward(self, grad=None, *args, **kwargs):
        """Compute a backwards pass with respect to itself. By default
        this function is called without any arguments and computing wrt to itself
        first. Gradient information is passed from children to their parents via
        the grad argument."""
        pass

    def __eq__(self, value):
        return self.data == value
