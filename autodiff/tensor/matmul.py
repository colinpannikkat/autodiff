from ..graph.operations.base import BinOp
from ..tensor import Tensor


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes matrix multiplication between two Tensors.

    If the dimensionality of a or b is 1, then an elementwise multiplication will be
    computed across rows (or columns).

    Batched matmul is not yet implemented.
    """

    assert a.shape[1] == b.shape[0]

    m, n, p = a.shape[0], a.shape[1], b.shape[1]

    if len(a.shape) == 1:
        return a * b
    elif len(b.shape) == 1:
        return b * a

    if len(a.shape) > 2 or len(b.shape) > 2:
        raise NotImplementedError("MatMul with Tensors above 2D is not yet supported.")

    res = Tensor((m, p))

    for j in range(m):
        for i in range(p):
            for k in range(n):
                res[j][i] += a[k][i] * b[j][k]

    return res


class MatMul(BinOp):

    @staticmethod
    def _forward(x: Tensor, y: Tensor) -> Tensor:
        return matmul(x, y)

    @staticmethod
    def _backward(inputs: list[Tensor]) -> list[Tensor]:
        """Compute derivative for the matrix multiplication of two Tensors:
        z = x @ y
        dz/dx = y
        dz/dy = x
        """
        raise NotImplementedError()

    @staticmethod
    def op(x, y):
        return MatMul.forward(x, y)


Tensor.__matmul__ = MatMul.op