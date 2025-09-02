from autodiff import Variable
from autodiff.config import enable_higher_order_gradients


class TestNeg():
    def test_neg_simple(self):
        x = Variable(5.0)
        y = -x
        assert y == -5.0

    def test_neg_chain(self):
        x = Variable(2.0)
        y = -(-x)
        assert y == 2.0

    def test_neg_gradient(self):
        x = Variable(3.0)
        y = -x
        y.backward()
        assert x.grad == -1.0

    def test_neg_zero(self):
        x = Variable(0.0)
        y = -x
        assert y == 0.0

    def test_neg_multiple_variables(self):
        x = Variable(4.0)
        z = Variable(-2.0)
        y = -x + -z
        assert y == -4.0 + 2.0

    def test_neg_backward_chain(self):
        x = Variable(1.5)
        y = -(-(-x))
        y.backward()
        assert x.grad == -1.0

    def test_double_neg(self):

        enable_higher_order_gradients()

        x = Variable(2)
        y = -x

        assert y == -2

        x.zero_grad()
        y.backward()

        print(x.grad)

        assert x.grad == -1

        x_grad = x.grad
        x.zero_grad()
        x_grad.backward()

        assert x.grad is None
