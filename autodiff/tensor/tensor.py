from typing import Iterable
from ..graph.node import Node
from ..exceptions import InvalidMethodError
import random


class Tensor(Node):

    def __init__(
        self,
        shape: Iterable[int],
        requires_grad: bool = True,
        zeros_like: bool = True
    ):

        super().__init__(requires_grad)

        self.shape = shape
        self.data = self.create_ndarray_from_shape(shape, zeros_like)

    def __len__(self):
        raise InvalidMethodError(f"Cannot call method `len` on type {type(self)}.")

    def __eq__(self, value):
        return self.data == value

    def __repr__(self):
        return f"Tensor({self.data}, grad={self.grad})"

    def __getitem__(self, i):
        return self.data[i]

    @property
    def T(self) -> "Tensor":

        if len(self.shape) != 2:
            raise InvalidMethodError("Transpose is only supported for 2D tensors.")

        transposed_data = [
            [self.data[i][j] for i in range(self.shape[0])]
            for j in range(self.shape[1])
        ]

        # Set the data directly to avoid re-initialization
        tensor = Tensor((self.shape[1], self.shape[0]), self.requires_grad, zeros_like=True)
        tensor.data = transposed_data
        return tensor

    @staticmethod
    def create_filled_list_from_length(
        length: int,
        zeros_like: bool = True
    ):
        """Creates a filled list from a given length. By default a list is
        filled with zeros.

        Inputs:

            length (int): Length of list
            zeros_like (bool): Whether to fill with zeros or random floating
            point numbers sampled from the Gaussian distribution.
        """

        if not zeros_like:
            return [random.normalvariate() for _ in range(length)]

        return [0.] * length

    @staticmethod
    def create_ndarray_from_shape(
        shape: Iterable[int],
        zeros_like: bool = True
    ):
        """Returns an nth dimensional array given a shape.

        Inputs:
            shape (Iterable): An iterable containing integers denoting the
            shape information for ndarray.
            zeros_like (bool): Flag to make zeros_like or random.

        Information is stored row-major.

        Each item denotes a dimension. For example passing
        `(512, 128)` denotes a two dimensional array with 512 rows
        and 128 columns.

        This function works by recursively building an nth dimensional array
        from the last dimension to the first. It is equivalent to the following
        iterative code.

        ```
        i_arr = []
        x, y, z, ... = shape
        for i in x:
            j_arr = []
            for j in y:
                k_arr = []
                for k in z:
                    # ...
                    k_arr.append(0.)
                j_arr.append(k_arr)
            i_arr.append(j_arr)
        ```
        """

        # Base cases
        match len(shape):
            case 0:
                raise Exception("Shape information is empty.")
            case 1:
                return Tensor.create_filled_list_from_length(shape[0], zeros_like)
            case _:
                pass

        n1 = shape[0]
        data = []

        # For each index in n1, add an (n2, ...) array to it
        for _ in range(n1):
            data.append(Tensor.create_ndarray_from_shape(shape[1:], zeros_like))

        return data

    def backward(self, grad=None, *args, **kwargs):
        return super().backward(grad, *args, **kwargs)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
