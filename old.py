def train_perceptron(inputs, real_outputs, alpha=1):
    weights = [0] * len(inputs[0])
    bias = 0

    training = True
    while training:
        training = False
        for i, input in enumerate(inputs):
            error = calc_error(input, real_outputs[i], weights, bias)
            if error:
                for j, item in enumerate(input):
                    training = True
                    weights[j] += item * real_outputs[i] * alpha
                bias += real_outputs[i] * alpha
   
    return weights, bias

def calc_error(input, real_output, weights, bias):
    pre_output = bias

    for idx, item in enumerate(input):
        pre_output += item * weights[idx]

    if pre_output >= 0:
        output = 1
    else:
        output = -1

    error = real_output - output
    return error

def validate_perceptron(input, real_outputs, weights, bias):
    for i, vetor in enumerate(input):
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


def train_multi_outputs_perceptron(inputs, alpha=1):
    weights = []
    
    for idx in range(len(inputs)):
        outputs = [-1] * (len(inputs) - 1)
        outputs.insert(idx, 1)
        
        result = train_perceptron(inputs, outputs, alpha)
        weights.append(result)

    return weights
    

def validate_multi_outputs_perceptrons(inputs, all_weights):
    
    for idx, (weights, bias)  in enumerate(all_weights):
        outputs = [-1] * (len(inputs) - 1)
        outputs.insert(idx, 1)

        if not validate_perceptron(inputs, outputs, weights, bias):
            return False

    return True

def check_multi_outputs_perceptrons(input, all_weights, truth_table):

    result = []
    for weights, bias in all_weights:

        pre_output = bias
    
        for i, item in enumerate(input):
            pre_output += item * weights[i]

        if pre_output >= 0:
            output = 1
        else:
            output = -1

        result.append(output)
    
    idx = result.index(1)
    return truth_table[idx]





X = [
     1, -1, -1, -1, 1,
    -1, 1, -1, 1, -1,
    -1, -1, 1, -1, -1,
    -1, 1, -1, 1, -1,
    1, -1, -1, -1, 1,
]
T = [
    1, 1, 1, 1, 1,
    -1, -1, 1, -1, -1,
    -1, -1, 1, -1, -1,
    -1, -1, 1, -1, -1,
    -1, -1, 1, -1, -1,
]
C = [
    1, 1, 1, 1, 1,
    1, -1, -1, -1, -1,
    1, -1, -1, -1, -1,
    1, -1, -1, -1, -1,
    1, 1, 1, 1, 1,
]


inputs = [X, T, C]

outputs = [1, -1] 

# weights, bias = train_perceptron(inputs, outputs)

# result = validate_perceptron(inputs, outputs, weights, bias)

# print(result)

weights = train_multi_outputs_perceptron(inputs)
valid = validate_multi_outputs_perceptrons(inputs, weights)
print(ALPHABET)

print(valid)

truth_table = ["X", "T", "C"]
result = check_multi_outputs_perceptrons(C, weights, truth_table)

print(result)




