
import numpy as np


class Tanh:
    def activate(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    def derivative(self, x):
        return 1 - self.activate(x) ** 2


class Layer:
    def __init__(self, len_inputs, neurons, function, last=False):
        shape = neurons, len_inputs + 1
        self.weights = np.random.uniform(-0.5, 0.5, size=shape)
        self.f = function
        self.last = last
        self.idx = None
        self.neurons = neurons
        self.len_inputs = len_inputs
    
    def forward(self, layer_input):
        self.input = layer_input
        self.net = self.input.dot(self.weights.T)
        self.output = self.f.activate(self.net)
        return self.output
    
    def backward(self, target, alpha, previous_delta=None, previous_weigth=None):
        if self.last:
            self.delta = (target - self.output) * self.f.derivative(self.net)
        else:
            self.delta = (np.delete(previous_delta.dot(previous_weigth).T, 0) * self.f.derivative(self.net))
        
        self.weights += self.delta.T * self.input * alpha
        
        return self.delta, self.weights
        
        
        
        
    def __repr__(self):
        return f"({self.idx}º Layer, Neurons: {self.neurons}, Last: {self.last})"


class NeuralNetwork:
    def __init__(self, *layers: Layer):
        self.layers = list(layers)
        for idx, layer in enumerate(self.layers):
            layer.idx = idx + 1
        self.layers[-1].last = True
        self.len_inputs = self.layers[0].len_inputs
        
    def __repr__(self):
        return f"NeuralNetwork (Num_Layers: {len(self.layers)}, Len_Inputs: {self.len_inputs}, Layers: {self.layers})"
    
    @property
    def weights(self):
        resp = []
        for idx, layer in enumerate(self.layers):
            resp.append((idx+1, layer.weights))
        return resp
        
    def _forward(self, x_input):
        #input_layer = x_input
        input_layer = np.array([np.append(1, x) for x in x_input])
        for layer in self.layers:
            out_layer = layer.forward(input_layer)
            input_layer = np.array([np.append(1, x) for x in out_layer])
            
        return out_layer
    
    def _backward(self, y, alpha):
        for layer in reversed(self.layers):
            if layer.last:
                previous_delta, previous_weigth = layer.backward(y, alpha)
            else:
                previous_delta, previous_weigth = layer.backward(y, alpha, previous_delta, previous_weigth)
    
    def fit(self, x_train, y_train, epochs=2000, alpha=0.05):
        for epoch in range(epochs):
            for x, y in zip(x_train, y_train):
                self._forward(x)
                self._backward(y, alpha)
                
            out = self._forward(x_train)
            errors = np.array([sum(error) for error in (y_train - out) ** 2])
            self.mean_squared_error = sum(errors) / len(errors)
            
            if not epoch % 100:
                print(f"MSE: {self.mean_squared_error}")
                
                
    def predict(self, x):
        out = self._forward(x)
        return out




x_train = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
y_train = [-.9602, -.5770, -.0729, .3771, .6405, .6600, .4609, .1336, -.2013, -.4344, -.5000 ]
x_train = np.array([[x] for x in x_train])
y_train = np.array([[y] for y in y_train])


rede = NeuralNetwork(
    Layer(len_inputs=1, neurons=4, function=Tanh()),
    Layer(len_inputs=4, neurons=1, function=Tanh()),
)
rede.fit(x_train, y_train)
out = rede.predict(x_train)


import matplotlib.pyplot as plt


plt.scatter(x_train, y_train, color="green")
plt.scatter(x_train, out, color="red")
plt.show()
