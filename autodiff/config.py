"""General global configuration"""


"""
The following are to enable and disable _higher_order_gradients, enabling higher
ordered gradients lead to significant performance decreases.
"""
_higher_order_gradients = False


def enable_higher_order_gradients():
    global _higher_order_gradients
    _higher_order_gradients = True


def disable_higher_order_gradients():
    global _higher_order_gradients
    _higher_order_gradients = False


def is_higher_order_gradients_enabled() -> bool:
    return _higher_order_gradients
