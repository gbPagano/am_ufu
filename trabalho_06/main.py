import polars as pl
import numpy as np

from model import Adaline

data = pl.read_excel("Basedados_B2.xlsx")

outputs = data.drop_in_place("t")
inputs = np.array([np.array(value) for value in data.rows()])

from rich.traceback import install; install()
ia = Adaline()

ia.fit(inputs, outputs, epochs=20)


