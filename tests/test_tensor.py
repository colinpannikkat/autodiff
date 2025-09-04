from autodiff.tensor import Tensor
from autodiff.exceptions import InvalidMethodError
import autodiff as ad
import pytest
import numpy as np  # used for matmul comparisons


class TestTensor:

    def test_1d_instantiation(self):

        n1 = 512

        x = Tensor((n1,))

        assert x.shape[0] == 512

        assert x == [0.] * 512

    def test_len_error(self):

        x = Tensor((1,))

        with pytest.raises(InvalidMethodError):
            len(x)

    def test_2d_instantiation(self):

        n1, n2 = 512, 128

        x = Tensor((n1, n2))

        assert x.shape[0] == n1
        assert x.shape[1] == n2

        assert x[0] == [0.] * n2
        assert x[0][0] == 0.

    def test_3d_instantiation(self):

        n1, n2, n3 = 512, 128, 64

        x = Tensor((n1, n2, n3))

        assert x.shape[0] == n1
        assert x.shape[1] == n2
        assert x.shape[2] == n3

        assert x[0] == [[0.] * n3] * n2
        assert x[0][0] == [0.] * n3
        assert x[0][0][0] == 0.

    def test_2d_matmul(self):

        x = Tensor((2, 2))

        # Fill with 1's on first row and 2's on second
        for i in range(2):
            x[0][i] = 1
        for i in range(2):
            x[1][i] = 2

        y = Tensor((2, 2))

        # Fill with 2's on first row and 1's on second
        for i in range(2):
            x[0][i] = 2
        for i in range(2):
            x[1][i] = 1

        res = ad.matmul(x, y)

        np_x = np.array(x.data)
        np_y = np.array(y.data)
        np_res = np.matmul(np_x, np_y)

        assert np.array_equal(np.array(res.data), np_res)

    def test_random_init(self):

        x = Tensor((1,), zeros_like=False)

        assert x[0] != 0.

    def test_row_vector(self):

        x = Tensor((1, 2))

        assert x.data == np.zeros((1, 2)).tolist()

    def test_column_vector(self):

        x = Tensor((2, 1))

        assert x.data == np.zeros((2, 1)).tolist()

    def test_transpose(self):

        x = Tensor((2, 1))

        xT = x.T

        assert x.shape[::-1] == xT.shape
