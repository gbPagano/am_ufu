import pandas as pd


def get_data():
    data = pd.read_csv("train.csv")

    outputs = list(data.label)
    raw_inputs = data.drop("label", axis="columns")

    inputs = [[-1 if not x else 1 for x in line] for _, line in raw_inputs.iterrows()]

    return inputs, outputs

import polars as pl


def get_data_2():
    data = pl.read_csv("train.csv")

    outputs = list(data.drop_in_place("label")) 
    inputs = [[-1 if not x else 1 for x in line] for line in data.rows()]

    return inputs, outputs



from timeit import timeit

print("pandas:", timeit('get_data()', 'from __main__ import get_data', number=3))
print("polars:", timeit('get_data_2()', 'from __main__ import get_data_2', number=3))


