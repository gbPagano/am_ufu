
class Perceptron:
    def __init__(self, alpha=1):
        self.weights = []
        self.alpha = alpha


    def fit(self, inputs, raw_outputs):
        self.iterations = 0
        self.inputs = inputs
        self.outputs = raw_outputs

        for i in range(len(self.inputs)):
            outputs = [-1] * 9
            idx = self.outputs[i]
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
        try:
            return result.index(1)
        except ValueError:
            return "Failed"


    def _validate_perceptron(self):
        self.acertos = 0
        self.total= 0
        for idx, (weights, bias) in enumerate(self.weights):
            outputs = [-1] * len(self.inputs)
            outputs.insert(idx, 1)

            self._validate_neuron(outputs, weights, bias)
        
        result = self.acertos/self.total

        print(f"Trained with {result*100}% of precision")

    def _validate_neuron(self, real_outputs, weights, bias):
        for i, vetor in enumerate(self.inputs):
            self.total += 1
            pre_output = bias
            idx = self.outputs[i]

            for j, item in enumerate(vetor):
                pre_output += item * weights[j]

            if pre_output >= 0:
                output = 1
            else:
                output = -1

            if output == real_outputs[idx]:
                self.acertos += 1



    def _train_neuron(self, real_outputs):
        weights = [0] * len(self.inputs[0])
        bias = 0
        
        training = True
        while training:
            print("iteration:", self.iterations)
            self.iterations += 1
            training = False
            for i, input in enumerate(self.inputs):
                idx = self.outputs[i]
                error = self._calc_error(input, real_outputs[idx], weights, bias)
                if error:
                    for j, item in enumerate(input):
                        training = True
                        weights[j] += item * real_outputs[idx] * self.alpha
                    bias += real_outputs[idx] * self.alpha
       
        return weights, bias

    def _calc_error(self, input, real_output, weights, bias):
        pre_output = bias

        for idx, item in enumerate(input):
            pre_output += item * weights[idx]

        if pre_output >= 0:
            output = 1
        else:
            output = -1

        return real_output != output







