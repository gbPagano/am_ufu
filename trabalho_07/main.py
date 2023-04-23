import matplotlib.pyplot as plt
import numpy as np

from model import Adaline


with open("data.txt") as file:
    data = [tuple(map(float, line.strip(" \n").split())) for line in file.readlines()]  # reading txt data file
x_vec, y_vec = list(map(np.array, zip(*data)))


# calculating Pearson's correlation coefficient and the coefficient of determination
n = len(x_vec)
pearson_numerator = n * sum(x_vec * y_vec) - sum(x_vec) * sum(y_vec)
pearson_denominator = np.sqrt(n * sum(x_vec ** 2) - sum(x_vec) ** 2) * np.sqrt(n * sum(y_vec ** 2) - sum(y_vec) ** 2)
corr_pearson = pearson_numerator / pearson_denominator
determination_coefficient = corr_pearson ** 2


# performing linear regression without adaline :(
b = (n * sum(x_vec * y_vec) - sum(x_vec) * sum(y_vec)) / (n * sum(x_vec ** 2) - sum(x_vec) ** 2)
a = y_vec.mean() - b * x_vec.mean()


# training adaline
model = Adaline()
model.fit(
    np.array([np.array([x]) for x in x_vec]),
    np.array([np.array([y]) for y in y_vec]),
)


# prints
print("Pearson Correlation:", corr_pearson)
print("Determination Coefficient:", determination_coefficient)
print("Linear variables:", [b, a])
print("Adaline linear variables:", model.weights)


# plots
x_plot = np.linspace(-5, 12, 200)
plt.figure(figsize=(15, 10)).suptitle('Approaching Linear Regression with Adaline')

# plotting linear regression without adaline :(
plt.subplot(221)
plt.plot(x_vec, y_vec, "go")  # plotting data dots
y_plot = x_plot*b + a
plt.plot(x_plot, y_plot, "red")  # plotting linear regression without adaline
plt.axis([-4, 11, -4, 11])
plt.xlabel("X")
plt.ylabel("Y", rotation=True)
plt.title("Linear Regression without Adaline :(")

# plotting linear regression with adaline :)
a_ia = model.weights[1]
b_ia = model.weights[0]
plt.subplot(222)
plt.plot(x_vec, y_vec, "go")  # plotting data dots
y_ia = x_plot*b_ia + a_ia
plt.plot(x_plot, y_ia, "blue")  # plotting linear regression with adaline
plt.axis([-4, 11, -4, 11])
plt.xlabel("X")
plt.ylabel("Y", rotation=True)
plt.title("Linear Regression with Adaline :)")

# plotting both linear regressions without dots
plt.subplot(223)
y_plot = x_plot*b + a
plt.plot(x_plot, y_plot, "red")  # plotting linear regression without adaline
y_ia = x_plot*b_ia + a_ia
plt.plot(x_plot, y_ia, "blue")  # plotting linear regression with adaline
plt.axis([-4, 11, -4, 11])
plt.xlabel("X")
plt.ylabel("Y", rotation=True)
plt.title("Both Linear Regressions")
plt.legend(['Without Adaline', 'With Adaline'])

# plotting model mean squared erros
plt.subplot(224)
plt.plot(model.errors)
plt.axis(ymin=0)
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.title("Adaline MSE over time")

# plot
plt.show()
