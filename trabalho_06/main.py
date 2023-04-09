import polars as pl
import numpy as np
import matplotlib.pyplot as plt

from model import Adaline

data = pl.read_excel("trabalho_06/Basedados_B2.xlsx")

outputs = data.drop_in_place("t")
inputs = np.array([np.array(value) for value in data.rows()])

from rich.traceback import install; install()
ia = Adaline()

ia.fit(inputs, outputs, epochs=200)


dots = list(zip(data.get_column("s1"), data.get_column("s2")))
green = [dot for dot, out in zip(dots, outputs) if out == 1]
red = [dot for dot, out in zip(dots, outputs) if out != 1]

green_x, green_y = list(zip(*green)) 
red_x, red_y = list(zip(*red))

plt.figure(figsize=(14,6))
plt.subplot(121)
x = np.arange(0., 5., 0.2)
plt.plot(x, (-x*ia.weights[0] - ia.weights[2]) / ia.weights[1])


plt.plot(green_x, green_y, "go")
plt.plot(red_x, red_y, "ro")
plt.axis([0, 3, 0, 3])


plt.subplot(122)
plt.plot(ia.errors)


plt.show()

