import numpy as np
from rich.progress import track

class Tanh:
    def activate(self, x):
        return np.tanh(x)
    
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
    
    def backward(self, alpha, previous_delta=None, previous_weigth=None, error=None):
        if self.last:
            self.delta = error * self.f.derivative(self.net)
        else:
            self.delta = (np.delete(previous_delta.dot(previous_weigth).T, 0) * self.f.derivative(self.net))
        
        self.weights += np.array([self.delta]).T * np.array([self.input]) * alpha
        
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
        self.all_mse = []
        
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
        input_layer = np.append(1, x_input)
        for layer in self.layers:
            out_layer = layer.forward(input_layer)
            input_layer = np.append(1, out_layer)
            
        return out_layer
    
    def _backward(self, alpha, error):
        for layer in reversed(self.layers):
            if layer.last:
                previous_delta, previous_weigth = layer.backward(alpha, error=error)
            else:
                previous_delta, previous_weigth = layer.backward(alpha, previous_delta, previous_weigth)
    
    def fit(self, x_train, y_train, epochs=2000, alpha=0.05, batch_size=1, print_mse=True):

        for epoch in track(range(epochs), description="Processing..."):
            outputs = []
            batch_errors = []
            data = list(zip(x_train,y_train))
            # np.random.shuffle(data)
            x_train,y_train = zip(*data)
            x_train,y_train = np.array(x_train), np.array(y_train)
            for x, y in zip(x_train, y_train):
                out = self._forward(x)
                error = (y - out)

                batch_errors.append(error)
                if len(batch_errors) == batch_size:
                
                    batch_error = sum(batch_errors) / batch_size
                    batch_errors = []
                    self._backward(alpha, batch_error)

                outputs.append(out)
                
            errors = np.array([sum(error) for error in (y_train - outputs) ** 2])
            self.mean_squared_error = sum(errors) / len(errors)
            self.all_mse.append(self.mean_squared_error)
            
            if not epoch % 100 and print_mse:
                print(f"MSE: {self.mean_squared_error}")

                
                
    def predict(self, x):
        out = self._forward(x)
        return out
    

import polars as pl


def number_to_neurons(n):
    res = [-1] * 10
    res[n] = 1
    return res

def evaluate(rede, x, y, total, inicial=0):
    points = 0
    for idx in range(inicial ,total):
        correct = np.argmax(y[idx])
        predict = np.argmax(rede.predict(x[idx]))
        if correct == predict:
            points += 1

    return points/total * 100



def save_weights(rede, n_cam):
    with open("weights.npy", "wb") as f:
        for idx in range(n_cam):
            np.save(f, rede.layers[idx].weights)

def load_weights(rede, n_cam):
    with open("weights.npy", "rb") as f:
        for idx in range(n_cam):
            rede.layers[idx].weights = np.load(f)

