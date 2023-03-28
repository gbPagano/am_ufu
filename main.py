

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


inputs = [
    [
        1, -1, -1, -1, 1,
        -1, 1, -1, 1, -1,
        -1, -1, 1, -1, -1,
        -1, 1, -1, 1, -1,
        1, -1, -1, -1, 1,
    ],
    [
        1, 1, 1, 1, 1,
        -1, -1, 1, -1, -1,
        -1, -1, 1, -1, -1,
        -1, -1, 1, -1, -1,
        -1, -1, 1, -1, -1,
    ]
]
outputs = [1, -1] 

weights, bias = train_perceptron(inputs, outputs)

result = validate_perceptron(inputs, outputs, weights, bias)

print(result)
