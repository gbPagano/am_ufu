import numpy as np
from rich.progress import track

class Adaline:
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.errors = []
        self.theta = 0

    def fit(self, inputs, outputs, epochs = 200):
        self.weights = np.random.uniform(-0.5, 0.5, len(inputs[0]) + 1)
        
        inputs = np.array([np.append(vector, 1) for vector in inputs])  # adding bias

        for _ in track(range(epochs), description="Processing..."):
            quadratic_error = 0
            for input, real_output in zip(inputs, outputs):
                net = sum(input * self.weights)
                
                predicted = net  # linear function
                
                quadratic_error += (real_output - predicted) ** 2

                self.weights += self.alpha * (real_output - predicted) * input
            
            
            # print(self.weights)
            self.errors.append(quadratic_error)

        # self.weights[2] += 0.2
        self._validate(inputs, outputs)


    def test(self, inputs, outputs):
        
        total = 0
        errors = 0
        for input, real_output in zip(inputs, outputs):
            net = sum(input * self.weights)
                
            predicted = 1 if net >= self.theta else -1  # linear function
            
            total += 1
            if predicted != real_output:
                errors += 1
                print(input)

        return (total - errors) / total * 100



    def _validate(self, inputs, outputs):
        result = self.test(inputs, outputs)
        
        print(f"Succesfully trained with: {result}% of precision")
