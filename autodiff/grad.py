from .graph import _cast_to_variable, Variable

def grad(func):
    """
    Returns a callable that computes the gradient of the given function `func`
    with respect to its inputs. Supports chaining (e.g., grad(grad(f))).
    """
    pass