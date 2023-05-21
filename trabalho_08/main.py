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



        self.weights += np.array([self.delta]).dot(np.array([self.input])) * alpha
        
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
    
    def fit(self, x_train, y_train, epochs=2000, alpha=0.05, batch_size=1):

        for epoch in track(range(epochs), description="Processing..."):
            outputs = []
            batch_errors = []
            data = list(zip(x_train,y_train))
            np.random.shuffle(data)
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
rede.fit(x_train, y_train, epochs=20, alpha=0.05)


import matplotlib.pyplot as plt

out = rede.predict(x_train)

plt.scatter(x_train, y_train, color="green")
plt.scatter(x_train, out, color="red")
plt.legend(['Official', 'Trained'])
plt.show()


plt.plot(rede.all_mse)
plt.show()
