from model import Perceptron
from digits import *


def get_data():
    outputs = [0,1,2,3,4,5,6,7,8,9] 
    inputs = NUMBERS

    return inputs, outputs


def get_data_2():
    outputs = [0,1,2,3,4,5,6,7,8,9] * 2
    inputs = NUMBERS + NUMBERS_ALT

    return inputs, outputs

# Treino com uma variação de numeros
inputs, outputs = get_data()

ia = Perceptron()

ia.fit(inputs, outputs)


# Treino com duas variações de numeros
inputs2, outputs2 = get_data_2()

ia2 = Perceptron()

ia2.fit(inputs2, outputs2)


