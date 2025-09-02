"""Test cases for computational graph functions"""
from autodiff import Variable, Node
from autodiff.func import exp
from autodiff.config import enable_higher_order_gradients, disable_higher_order_gradients
import pytest
import numpy as np


class TestGraph:
    def test_node_instantiation(self):
        with pytest.raises(TypeError):
            Node()

    def test_variable_forward(self):
        number = 3.
        var = Variable(number)
        assert var.forward() == number

    def test_add_operation(self):
        x = Variable(2.)
        y = Variable(3.)

        z = x + y

        assert z == 5.

    def test_mul_operation(self):
        x = Variable(2.)
        y = Variable(3.)

        z = x * y

        assert z == 6.

    def test_operation_with_non_variable(self):
        x = 2
        y = Variable(3)

        z = x * y

        assert z == 6.

    def test_operation_with_non_variable_right(self):
        x = 2
        y = Variable(3)

        z = y * x

        assert z == 6.

    def test_mul_backward(self):
        x = Variable(2)
        y = Variable(3)

        z = x * y
        z.backward()

        assert z.grad == 1
        assert x.grad == 3
        assert y.grad == 2

    def test_empty_backward(self):
        x = Variable(1.)
        x.backward()

        assert x.grad == 1.

    def test_polynomial_grad(self):
        x = Variable(-1)

        y = x*x + 2*x + 1

        y.backward()

        assert y == 0
        assert x.grad == 0

    def test_double_derivative(self):
        # f(x) = x^2, f'(x) = 2x, f''(x) = 2
        enable_higher_order_gradients()
        x = Variable(3.0)
        y = x * x
        y.backward()
        grad_x = x.grad

        assert isinstance(grad_x, Variable)

        # Now compute second derivative by differentiating grad_x w.r.t x
        x.zero_grad()
        grad_x.backward()
        second_derivative = x.grad

        disable_higher_order_gradients()

        assert grad_x == 6.0
        assert second_derivative == 2.0

    def test_double_derivative_exp(self):
        # f(x) = exp(x), f'(x) = exp(x), f''(x) = exp(x)
        enable_higher_order_gradients()
        x = Variable(1.0)
        y = exp(x)
        y.backward()
        grad_x = x.grad

        assert isinstance(grad_x, Variable)
        assert grad_x.requires_grad

        x.zero_grad()
        grad_x.backward()
        second_derivative = x.grad

        disable_higher_order_gradients()

        assert np.isclose(grad_x.data, np.exp(1.0))
        assert np.isclose(second_derivative.data, np.exp(1.0))

    def test_division(self):
        x = Variable(3)
        y = Variable(1)

        z = x*x / (y+3)

        assert z == 2.25

    def test_division_with_non_variable(self):
        x = 3
        y = Variable(1)

        z = x / (y + 3)

        assert z == 0.75

    def test_division_with_non_variable_right(self):
        x = Variable(3)
        y = 1

        z = x / (y + 3)

        assert z == 0.75

    def test_divison_backward(self):
        x = Variable(3)
        y = Variable(1)

        z = x*x / (y+3)

        z.backward()

        assert x.grad == 3/2
        assert y.grad == -9/16

    def test_divison_same_var(self):
        x = Variable(3)

        z = (x*x*x) / (x*x)

        z.backward(retain_graph=True)

        assert x.grad == 1

    def test_sub_operation(self):
        x = Variable(5.)
        y = Variable(3.)

        z = x - y

        assert z == 2.

    def test_sub_operation_with_non_variable(self):
        x = 5
        y = Variable(3)

        z = x - y

        assert z == 2.

    def test_sub_operation_with_non_variable_right(self):
        x = 5
        y = Variable(3)

        z = y - x

        assert z == -2.

    def test_sub_backward(self):
        x = Variable(5.)
        y = Variable(3.)

        z = x - y
        z.backward()

        assert z.grad == 1
        assert x.grad == 1
        assert y.grad == -1

    def test_exp_operation(self):
        x = Variable(2.)
        z = exp(x)
        assert np.isclose(z.data, np.exp(2.))

    def test_exp_backward(self):
        x = Variable(1.)

        y = exp(x)

        y.backward()

        assert np.isclose(y.data, np.exp(1))

    def test_exp_chain_rule(self):
        # y = exp(x^2)
        x = Variable(2.)
        y = exp(x*x)

        y.backward()

        true_grad = 2 * x.data * np.exp(x.data ** 2)
        print(true_grad)

        assert np.isclose(x.grad, true_grad)

    def test_power_operation(self):
        x = Variable(2.)
        y = Variable(3.)
        z = x ** y
        assert np.isclose(z.data, 8.0)

    def test_power_with_non_variable(self):
        x = 2
        y = Variable(3)
        z = x ** y
        assert np.isclose(z.data, 8.0)

    def test_power_with_non_variable_right(self):
        x = Variable(2)
        y = 3
        z = x ** y
        assert np.isclose(z.data, 8.0)

    def test_power_backward(self):
        x = Variable(2.)
        y = Variable(3.)
        z = x ** y
        z.backward()
        # dz/dx = y * x^(y-1) = 3 * 2^(2) = 12
        # dz/dy = x^y * log(x) = 8 * log(2)
        assert np.isclose(x.grad, 12.0)
        assert np.isclose(y.grad, 8.0 * np.log(2.0))

    def test_power_scalar_exponent_backward(self):
        x = Variable(4.)
        z = x ** 2
        z.backward()
        # dz/dx = 2 * x^(2-1) = 2 * 4 = 8
        assert np.isclose(x.grad, 8.0)

    def test_power_zero_exponent(self):
        x = Variable(5.)
        z = x ** 0
        assert np.isclose(z.data, 1.0)

    def test_power_zero_base(self):
        x = Variable(0.)
        y = Variable(3.)
        z = x ** y
        assert np.isclose(z.data, 0.0)

    def test_power_chain_rule(self):
        # y = (x^2)^3 = x^6
        x = Variable(2.)
        x2 = x ** 2
        y = x2 ** 3
        y.backward()
        # dy/dx = 6 * x^5 = 6 * 32 = 192
        assert np.isclose(x.grad, 192.0)

    def test_power_zero_backward(self):
        x = Variable(0)
        y = x ** 3

        y.backward()

        assert y == 0
        assert x.grad == 0
