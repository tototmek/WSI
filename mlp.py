from math import exp
import random
import numpy as np

def sigmoid(x):
    return 1 / (1 + exp(-x))

def d_sigmoid_dt(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return (0 if x < 0 else x)

def d_relu_dt(x):
    return (0 if x < 0 else 1)

def leakyrelu(x):
    return (0.1 * x if x < 0 else x)

def d_leakyrelu_dt(x):
    return (0.1 if x < 0 else 1)

def elu(x):
    return (exp(x) - 1 if x < 0 else x)

def d_elu_dt(x):
    return (exp(x) if x < 0 else 1)

def neuron(inputs, weights, bias, activation):
    return activation(sum([i * w for i, w in zip(inputs, weights)]) + bias)

class MLP:
    def __init__(self, input_size, hidden_size, n_hidden, output_size, activation, d_activation_dt):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.output_size = output_size
        self.activation = activation
        self.d_activation_dt = d_activation_dt
        self.weights = []
        self.biases = []
        self._init()

    def _init(self):
        for i in range(self.n_hidden + 1):
            if i == 0:
                self.weights.append([[random.random() for _ in range(self.input_size)] for _ in range(self.hidden_size)])
                self.biases.append([random.random() for _ in range(self.hidden_size)])
            elif i == self.n_hidden:
                self.weights.append([[random.random() for _ in range(self.hidden_size)] for _ in range(self.output_size)])
                self.biases.append([random.random() for _ in range(self.output_size)])
            else:
                self.weights.append([[random.random() for _ in range(self.hidden_size)] for _ in range(self.hidden_size)])
                self.biases.append([random.random() for _ in range(self.hidden_size)])
        
    def forward(self, inputs):
        outputs = [inputs]
        for i in range(self.n_hidden + 1):
            outputs.append([neuron(outputs[i], self.weights[i][j], self.biases[i][j], self.activation) for j in range(len(self.weights[i]))])
        return outputs
    
    def backward(self, outputs, targets, learning_rate):
        deltas = []
        for i in range(self.n_hidden + 1, 0, -1):
            if i == self.n_hidden + 1:
                deltas.append([outputs[i][j] - targets[j] for j in range(len(outputs[i]))])
            else:
                deltas.append([sum([deltas[-1][k] * self.weights[i][k][j] for k in range(len(deltas[-1]))]) * self.d_activation_dt(outputs[i][j]) for j in range(len(outputs[i]))])

        deltas.reverse()
        for i in range(self.n_hidden + 1):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] -= learning_rate * deltas[i][j] * outputs[i][k]
                self.biases[i][j] -= learning_rate * deltas[i][j]
    
    def fit(self, X, y, epochs, learning_rate, validation_data=None):
        self.weights = []
        self.biases = []
        history = {"train": [], "val": []}
        self._init()
        for _ in range(epochs):
            for i in range(len(X)):
                outputs = self.forward(X[i])
                self.backward(outputs, y[i], learning_rate)
            if validation_data:
                y_pred = self.predict(X)
                history["train"].append(accuracy(np.argmax(y, axis=1), np.argmax(y_pred, axis=1)))
                X_val, y_val = validation_data
                y_pred = self.predict(X_val)
                history["val"].append(accuracy(np.argmax(y_val, axis=1), np.argmax(y_pred, axis=1)))
        return history
    
    def predict(self, X):
        return [self.forward(x)[-1] for x in X]
    
    def cross_validate(self, folds, X, y, epochs, learning_rate):
        fold_size = len(X) // folds
        scores = []
        X = list(X)
        y = list(y)
        for i in range(folds):
            X_train = X[:i * fold_size] + X[(i + 1) * fold_size:]
            y_train = y[:i * fold_size] + y[(i + 1) * fold_size:]
            X_test = X[i * fold_size:(i + 1) * fold_size]
            y_test = y[i * fold_size:(i + 1) * fold_size]
            self.fit(X_train, y_train, epochs, learning_rate)
            y_pred = self.predict(X_test)
            scores.append(accuracy(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)))
        return scores
            


def accuracy(y_target, y_pred):
    assert len(y_target) == len(y_pred)
    score = 0
    for i, y in enumerate(y_target):
        if y == y_pred[i]:
            score += 1
    return score / len(y_target)
