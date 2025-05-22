import numpy as np
from .graph import UnaryOp, BinOp, Node, Variable, _cast_to_variable

# Max
class Max(BinOp):
    def __repr__(self):
        return "<Max>"

    def forward(x, y):
        x, y = _cast_to_variable(x, y)
        return Variable(np.maximum(x.data, y.data), [x, y], op=Max(x, y))

    def backward(self, grad):
        mask_x = self.x.data >= self.y.data
        mask_y = self.y.data > self.x.data
        dx = grad * mask_x
        dy = grad * mask_y
        return dx, dy

def max(x, y):
    return Max.forward(x, y)

# ReLU
class ReLU(UnaryOp):
    def __repr__(self):
        return "<ReLU>"

    def forward(x):
        x = _cast_to_variable(x)
        return Variable(np.maximum(x.data, 0), [x], op=ReLU(x))

    def backward(self, grad):
        mask = self.x.data > 0
        return grad * mask

def relu(x):
    return ReLU.forward(x)

# Sigmoid
class Sigmoid(UnaryOp):
    def __repr__(self):
        return "<Sigmoid>"

    def forward(x):
        x = _cast_to_variable(x)
        data = 1 / (1 + np.exp(-x.data))
        return Variable(data, [x], op=Sigmoid(x))

    def backward(self, grad):
        sigmoid_data = 1 / (1 + np.exp(-self.x.data))
        return grad * sigmoid_data * (1 - sigmoid_data)

def sigmoid(x):
    return Sigmoid.forward(x)

# Softmax
class Softmax(UnaryOp):
    def __repr__(self):
        return "<Softmax>"

    def forward(x):
        x = _cast_to_variable(x)
        shifted = x.data - np.max(x.data, axis=1, keepdims=True)
        exps = np.exp(shifted)
        softmax_data = exps / np.sum(exps, axis=1, keepdims=True)
        return Variable(softmax_data, [x], op=Softmax(x))

    def backward(self, grad):
        s = np.exp(self.x.data - np.max(self.x.data, axis=1, keepdims=True))
        s = s / np.sum(s, axis=1, keepdims=True)

        batch_size, _ = s.shape
        grad_input = np.zeros_like(grad)

        for i in range(batch_size):
            si = s[i].reshape(-1, 1)
            jacobian = np.diagflat(si) - si @ si.T
            grad_input[i] = jacobian @ grad[i]

        return grad_input

def softmax(x):
    return Softmax.forward(x)

# CrossEntropyLoss
class CrossEntropyLoss(BinOp):
    def __repr__(self):
        return "<CrossEntropyLoss>"
    
    def forward(predicted, target):
        predicted = _cast_to_variable(predicted)
        target = _cast_to_variable(target)

        shifted_logits = predicted.data - np.max(predicted.data, axis=1, keepdims=True)
        log_probs = shifted_logits - np.log(np.sum(np.exp(shifted_logits), axis=1, keepdims=True))
        loss = -np.sum(target.data * log_probs) / predicted.data.shape[0]

        return Variable(loss, [predicted, target], op=CrossEntropyLoss(predicted, target))

    def backward(self, grad):
        batch_size = self.x.data.shape[0]

        softmax_pred = np.exp(self.x.data - np.max(self.x.data, axis=1, keepdims=True))
        softmax_pred /= np.sum(softmax_pred, axis=1, keepdims=True)

        dx = grad * (softmax_pred - self.y.data) / batch_size
        dy = np.zeros_like(self.y.data)
        return dx, dy

def cross_entropy_loss(predicted, target):
    return CrossEntropyLoss.forward(predicted, target)

# Exp
class Exp(UnaryOp):
    
    def __repr__(self):
        return "<Exp>"

    def forward(x):
        x = _cast_to_variable(x)
        return Variable(np.exp(x.data), [x], op=Exp(x))
    
    def backward(self, grad):
        return grad * -np.exp(self.x.data)
    
def exp(x):
    return Exp.forward(x)