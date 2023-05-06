import numpy as np


class Tanh:
    def activate(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    def derivative(self, x):
        return 1 - self.activate(x) ** 2


class Layer:
    def __init__(self, neurons, len_inputs, function):
        shape = neurons, len_inputs
        self.weights = np.random.uniform(-0.5, 0.5, size=shape)
        self.f = function
    
    def forward(self, layer_input):
        self.input = layer_input
        self.net = self.input.dot(self.weights.T)
        self.output = self.f.activate(self.net)
        return self.output


class NeuralNetwork:
    def __init__(self, *layers):
        self.layers = list(layers)
        
    def forward(self, x_input):
        #input_layer = x_input
        input_layer = np.array([np.append(-1, x) for x in x_input])
        for layer in self.layers:
            out_layer = layer.forward(input_layer)
            input_layer = np.array([np.append(-1, x) for x in out_layer])
            
        return out_layer


weights_1 = np.array([
    [0.2, .4, .5],
    [.3, .6, .7],
    [.4, .8, .3],
])
weights_2 = np.array([
    [-.7, .6, .2, .7],
    [-.3, .7, .2, .8],
])
weights_3 = np.array([
    [.1, .8, .5],
])

x_input = np.array([
    [0.3, 0.7],
    [0.6, 1.4],
])


rede = NeuralNetwork(
    Layer(neurons=3, len_inputs=3, function=Tanh()),
    Layer(neurons=2, len_inputs=4, function=Tanh()),
    Layer(neurons=1, len_inputs=3, function=Tanh()),
)
rede.layers[0].weights = weights_1
rede.layers[1].weights = weights_2
rede.layers[2].weights = weights_3

res = rede.forward(x_input)
print(res)
