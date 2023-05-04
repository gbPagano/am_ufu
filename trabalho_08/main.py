import numpy as np



x_train = np.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
y_train = [-.9602, -.5770, -.0729,  .3771,  .6405,  .6600,  .4609, .1336, -.2013, -.4344, -.5000 ];
x_train = np.array([np.array([1, x]) for x in x_train])
y_train = np.array([np.array([y]) for y in y_train])



class NeuralNetwork:
    def __init__(self, x_train, y_train, alpha=0.01):
        self.alpha = alpha
        self.x = x_train
        self.d = y_train
        self.weights_1 = np.random.randn(2, len(x_train[0])) # first layer
        self.weights_2 = np.random.randn(1, len(self.weights_1[0]) + 1) # second layer
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def fit(self, epochs=500):

        for _ in range(epochs):
            # feedfowarding
            net_1 = np.dot(self.x, self.weights_1.T)
            output_1 = [self.sigmoid(u) for u in net_1]
            input_2 = np.array([np.append(1, u) for u in output_1])
            net_2 = np.dot(input_2, self.weights_2.T)
            output_2 = [self.sigmoid(u) for u in net_2]

            # backpropagation
            delta_2 = (self.d - output_2) * self.sigmoid_derivative(net_2)
            self.weights_2 += sum(self.alpha * delta_2 * input_2)
            
            delta_1 = [[-sum(x)] for x in delta_2 * self.weights_2] * self.sigmoid_derivative(net_1)
            self.weights_1 += sum(self.alpha * delta_1 * self.x)

        
    def predict(self, x):
        net_1 = np.dot(x, self.weights_1.T)
        output_1 = [self.sigmoid(u) for u in net_1]
        input_2 = np.array([np.append(1, u) for u in output_1])
        net_2 = np.dot(input_2, self.weights_2.T)
        output_2 = [self.sigmoid(u) for u in net_2]

        return output_2



    


model = NeuralNetwork(x_train, y_train)











