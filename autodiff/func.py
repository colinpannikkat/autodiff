from autodiff.graph.operations.math import Exp


def exp(x):
    return Exp.forward(x)
