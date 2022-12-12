from math import exp

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

def neuron(weight_list, input_list, activation):
    output = 0
    for i, input in enumerate(input_list):
        output += input * weight_list[i]
    return activation(output)

