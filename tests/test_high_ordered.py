from autodiff.config import enable_higher_order_gradients
from autodiff import Variable
from autodiff.func import exp
from autodiff.graph.print import print_graph
from autodiff.graph.operations import Pow
import numpy as np
from math import e
import torch


class TestHigherOrdered:

    enable_higher_order_gradients()

    def test_power_higher_ordered(self):

        enable_higher_order_gradients()

        x = Variable(5.)

        z = x ** 6 + 4 * x ** 4 + 5 * x ** 3 + x ** 2 + 5 * x

        assert z == 18800

        x.zero_grad()
        z.backward()

        assert x.grad == 21140  # first

        x_grad = x.grad
        x.zero_grad()
        x_grad.backward()

        assert x.grad == 20102  # second

        x_grad = x.grad
        x.zero_grad()
        x_grad.backward()

        assert x.grad == 15510  # third

        x_grad = x.grad
        x.zero_grad()
        x_grad.backward()

        assert x.grad == 9096  # fourth

        x_grad = x.grad
        x.zero_grad()
        x_grad.backward()

        assert x.grad == 3600  # fifth

        x_grad = x.grad
        x.zero_grad()
        x_grad.backward()

        assert x.grad == 720  # sixth

        x_grad = x.grad
        x.zero_grad()
        x_grad.backward()

        assert x.grad == 0  # seventh

        x_grad = x.grad
        x.zero_grad()
        x_grad.backward()

        assert x.grad == 0  # eighth

    def test_exponential_simple_no_neg(self):

        x = Variable(0.5)

        def exp(x):
            return Pow.forward(e, x)

        z = exp(x)
        ground = np.exp(x.data)

        assert np.isclose(z.data, ground)

        x.zero_grad()
        z.backward(retain_graph=True)

        # print_graph(x.grad)

        assert np.isclose(x.grad.data, ground)  # first

        x_grad = x.grad
        x.zero_grad()
        x_grad.backward(retain_graph=True)

        # print_graph(x.grad)

        assert np.isclose(x.grad.data, ground)  # second

    def test_exponential_simple_first_neg(self):

        x = Variable(0.5)

        z = exp(-x)
        ground = np.exp(-0.5)

        assert np.isclose(z.data, ground)

        x.zero_grad()
        z.backward(retain_graph=True)

        # print_graph(x.grad)

        assert np.isclose(x.grad.data, -ground)  # first

        x_grad = x.grad
        x.zero_grad()
        x_grad.backward(retain_graph=True)

        # print_graph(x.grad)

        assert np.isclose(x.grad.data, ground)  # second

    def test_exponential_simple_second_neg(self):

        x = Variable(0.5)

        z = -exp(x)
        ground = -np.exp(x.data)

        assert np.isclose(z.data, ground)

        x.zero_grad()
        z.backward(retain_graph=True)

        # print_graph(x.grad)

        assert np.isclose(x.grad.data, ground)  # first

        x_grad = x.grad
        x.zero_grad()
        x_grad.backward(retain_graph=True)

        # print_graph(x.grad)

        assert np.isclose(x.grad.data, ground)  # second

    def test_exponential_simple_higher_ordered(self):

        x = Variable(0.5)

        def exp(x):
            return Pow.forward(e, x)

        z = -exp(-x)

        assert np.isclose(z.data, -0.606530659713)

        x.zero_grad()
        z.backward(retain_graph=True)

        print_graph(x.grad)

        assert np.isclose(x.grad.data, 0.60653065971)  # first

        x_grad = x.grad
        x.zero_grad()
        x_grad.backward(retain_graph=True)

        print_graph(x.grad)

        assert np.isclose(x.grad.data, -0.60653065971)  # second

        x_grad = x.grad
        x.zero_grad()
        x_grad.backward()

        assert np.isclose(x.grad.data, 0.60653065971)  # third

    def test_exponential_higher_ordered(self):

        enable_higher_order_gradients()

        x = Variable(0.5)
        # z = (exp(x) - exp(-x)) / exp(x ** 2)
        # z = (exp(x) - exp(-x)) / x
        # z = exp(x) / x
        z = 1 / x

        # PyTorch comparison using autograd.grad for higher order derivatives
        xt = torch.tensor(0.5, requires_grad=True)
        # zt = (torch.exp(xt) - torch.exp(-xt)) / torch.exp(xt ** 2)
        # zt = (torch.exp(xt) - torch.exp(-xt)) / xt
        # zt = torch.exp(xt) / xt
        zt = 1 / xt
        assert np.isclose(z.data, zt.item())

        grads = []
        current = zt
        for i in range(3):
            grad = torch.autograd.grad(current, xt, retain_graph=True, create_graph=True)[0]
            grads.append(grad.item())
            current = grad

        autodiff_grads = []
        x.zero_grad()
        z.backward(retain_graph=True)
        autodiff_grads.append(x.grad.data)
        x_grad = x.grad
        for i in range(2):
            x.zero_grad()
            x_grad.backward(retain_graph=True)
            autodiff_grads.append(x.grad.data)
            x_grad = x.grad

        print(autodiff_grads)
        print(grads)

        for i in range(3):
            print(i + 1)
            assert np.isclose(autodiff_grads[i], grads[i])

    def test_neg_one_power(self):

        x = Variable(2.)

        z = (-x) ** 2

        assert np.isclose(z.data, 4)

        x.zero_grad()
        z.backward(retain_graph=True)

        assert np.isclose(x.grad.data, 4)  # first

    def test_neg_inside_power(self):

        x = Variable(-2.)

        z = (x) ** 2

        assert np.isclose(z.data, 4)

        x.zero_grad()
        z.backward(retain_graph=True)

        assert np.isclose(x.grad.data, -4)  # first

    def test_neg_high_power(self):

        enable_higher_order_gradients()

        x = Variable(2.)

        z = (-x) ** 6

        assert np.isclose(z.data, 64)

        x.zero_grad()
        z.backward(retain_graph=True)

        assert np.isclose(x.grad.data, 192)  # first

        x_grad = x.grad
        x.zero_grad()
        x_grad.backward(retain_graph=True)

        print_graph(x.grad)

        assert np.isclose(x.grad.data, 480)  # second

        x_grad = x.grad
        x.zero_grad()
        x_grad.backward(retain_graph=True)

        assert np.isclose(x.grad.data, 960)  # third

        x_grad = x.grad
        x.zero_grad()
        x_grad.backward(retain_graph=True)

        assert np.isclose(x.grad.data, 1440)  # fourth
