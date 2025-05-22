from autodiff import Variable
import autodiff as ad
from autodiff.func import relu, cross_entropy_loss
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

"""
A simple two-layer neural network for classification.

Used for testing autodifferentiation framework.

The NN trains on MNIST, which has 56000 training examples, and 10 classes.
"""

class TwoLayerNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = Variable.random((input_size, hidden_size)) * ((2 / input_size) ** 0.5)
        self.b1 = Variable.zeros((1, hidden_size))
        self.W2 = Variable.random((hidden_size, output_size)) * ((2 / hidden_size) ** 0.5)
        self.b2 = Variable.zeros((1, output_size))

    def forward(self, x):
        z1 = relu(x @ self.W1 + self.b1)
        z2 = z1 @ self.W2 + self.b2
        return z2
    
    def train(self, x_train, y_train, batch=128, lr=0.01, epochs=100):
        num_batches = x_train.shape[0] // batch
        for epoch in range(epochs):
            tot_loss = Variable(0)
            for i in range(num_batches):

                start = i * batch
                end = start + batch
                x_batch = x_train[start:end]
                y_batch = y_train[start:end]
                
                predictions = self.forward(x_batch)
                
                # Compute loss (cross-entropy)
                loss = cross_entropy_loss(predictions, y_batch)
                loss.backward()

                tot_loss += loss
                
                # Update weights and biases using SGD
                self.W1.data -= lr * self.W1.grad
                self.b1.data -= lr * self.b1.grad.sum(axis=0)
                self.W2.data -= lr * self.W2.grad
                self.b2.data -= lr * self.b2.grad.sum(axis=0)
                
                # Zero gradients
                self.W1.zero_grad()
                self.b1.zero_grad()
                self.W2.zero_grad()
                self.b2.zero_grad()

                del x_batch
                del y_batch
                del predictions
                del loss
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Loss: {tot_loss.data / num_batches}")

if __name__ == "__main__":

    mnist = fetch_openml("mnist_784", version=1)
    X, y = mnist.data / 255.0, mnist.target.astype(int)

    encoder = OneHotEncoder(categories="auto", sparse_output=False)
    y_onehot = encoder.fit_transform(y.to_numpy().reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y_onehot, test_size=0.2, random_state=42)
    X_train = Variable(X_train)
    y_train = Variable(y_train)

    input_size = 784
    hidden_size = 128
    output_size = 10
    nn = TwoLayerNN(input_size, hidden_size, output_size)
    nn.train(X_train, y_train, batch=512, lr=0.1, epochs=300)

    X_test = Variable(X_test)
    predictions = nn.forward(X_test)
    predicted_labels = predictions.data.argmax(axis=1)
    true_labels = y_test.argmax(axis=1)

    accuracy = (predicted_labels == true_labels).mean()
    print(f"Test Accuracy: {accuracy * 100:.2f}%")