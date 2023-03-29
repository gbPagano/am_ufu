from rich import print

from letters import ALPHABET


class Perceptron:
    def __init__(self, alpha=1):
        self.weights = []
        self.alpha = alpha


    def fit(self, inputs, reference_table):
        self.inputs = inputs
        self.reference_table = reference_table

        for idx in range(len(self.inputs)):
            outputs = [-1] * len(self.inputs)
            outputs.insert(idx, 1)
            
            result = self._train_neuron(outputs)
            self.weights.append(result)
        
        self._validate_perceptron()
    

    def predict(self, input):
        result = []
        for weights, bias in self.weights:

            pre_output = bias
        
            for i, item in enumerate(input):
                pre_output += item * weights[i]

            if pre_output >= 0:
                output = 1
            else:
                output = -1

            result.append(output)
        
        idx = result.index(1)
        return self.reference_table[idx]



    def _validate_perceptron(self):
        for idx, (weights, bias) in enumerate(self.weights):
            outputs = [-1] * len(self.inputs)
            outputs.insert(idx, 1)

            if not self._validate_neuron(outputs, weights, bias):
                raise Exception("It was not possible to train the neuron, possibly it is not a linearly solvable problem")

    def _validate_neuron(self, real_outputs, weights, bias):
        for i, vetor in enumerate(self.inputs):
            pre_output = bias
        
            for idx, item in enumerate(vetor):
                pre_output += item * weights[idx]

            if pre_output >= 0:
                output = 1
            else:
                output = -1

            if output != real_outputs[i]:
                return False

        return True


    def _train_neuron(self, real_outputs):
        weights = [0] * len(self.inputs[0])
        bias = 0

        training = True
        while training:
            training = False
            for i, input in enumerate(self.inputs):
                error = self._calc_error(input, real_outputs[i], weights, bias)
                if error:
                    for j, item in enumerate(input):
                        training = True
                        weights[j] += item * real_outputs[i] * self.alpha
                    bias += real_outputs[i] * self.alpha
       
        return weights, bias

    def _calc_error(self, input, real_output, weights, bias):
        pre_output = bias

        for idx, item in enumerate(input):
            pre_output += item * weights[idx]

        if pre_output >= 0:
            output = 1
        else:
            output = -1

        error = real_output - output
        return error




