import numpy as np


class Adaline:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.errors = []

    def fit(self, inputs, outputs, epochs = 200):
        self.weights = np.random.uniform(-0.5, 0.5, len(inputs[0]) + 1)
        
        inputs = np.array([np.append(vector, 1) for vector in inputs])  # adding bias

        for _ in range(epochs):
            quadratic_error = 0
            for input, real_output in zip(inputs, outputs):
                net = sum(input * self.weights)
                
                predicted = net  # linear function
                
                quadratic_error += (real_output - predicted) ** 2

                self.weights += self.alpha * (real_output - predicted) * input

            self.errors.append(quadratic_error)

            
