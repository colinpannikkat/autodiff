import autodiff as ad
from autodiff.func import exp
import matplotlib.pyplot as plt
import numpy as np

def tanh(x):
    return ((-exp((x * -2))) + 1) / (exp(-(x * 2)) + 1.0)

x = ad.Variable(1)
z = tanh(x)

z.backward()

print(z)
print(x.grad)

x = np.linspace(-7, 7, 700)
x_var = ad.Variable(x)
y = tanh(x_var)
y.backward()
grad = x_var.grad

plt.plot(   
            x, y.data,
            x, grad
        )                
plt.show()