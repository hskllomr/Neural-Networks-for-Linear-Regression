import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
import pandas as pd
import w3_tools

%matplotlib inline

np.random.seed(3)

import w3_unittest

m = 30

X, Y = make_regression(n_samples=m, n_features=1, noise=20, random_state=1)

X = X.reshape((1, m))
Y = Y.reshape((1, m))

print('Training dataset X:')
print(X)
print('Training dataset Y')
print(Y)


#veri kümesini çiz
plt.scatter(X,  Y, c="black")

plt.xlabel("$x$")
plt.ylabel("$y$")



shape_X = X.shape
shape_Y = Y.shape
m = X.shape[1]

print ('The shape of X: ' + str(shape_X))
print ('The shape of Y: ' + str(shape_Y))
print ('I have m = %d training examples!' % (m))

w3_unittest.test_shapes(shape_X, shape_Y, m)


def layer_sizes(X, Y):

    #giriş büyüklüğü.
    n_x = X.shape[0]
    #çıkış büyüklüğü.
    n_y = Y.shape[0]
    return (n_x, n_y)


(n_x, n_y) = layer_sizes(X, Y)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the output layer is: n_y = " + str(n_y))

w3_unittest.test_layer_sizes(layer_sizes)


def initialize_parameters(n_x, n_y):

    W = np.random.randn(n_y, n_x) * 0.01
    b = np.zeros((n_y, 1))

    assert (W.shape == (n_y, n_x))
    assert (b.shape == (n_y, 1))

    parameters = {"W": W,
                  "b": b}

    return parameters

parameters = initialize_parameters(n_x, n_y)
print("W = " + str(parameters["W"]))
print("b = " + str(parameters["b"]))


def forward_propagation(X, parameters):

    W = parameters.get("W")
    b = parameters.get("b")

    Z = np.matmul(W, X) + b
    #Z=wX+b
    #Y=Z
    Y_hat = Z

    assert (Y_hat.shape == (n_x, X.shape[1]))

    return Y_hat

Y_hat = forward_propagation(X, parameters)

print(Y_hat)


def compute_cost(Y_hat, Y):

    m = Y.shape[1]

    cost = np.sum((Y_hat - Y) ** 2) / (2 * m)

    return cost

print("cost = " + str(compute_cost(Y_hat, Y)))

parameters = w3_tools.train_nn(parameters, Y_hat, X, Y)

print("W = " + str(parameters["W"]))
print("b = " + str(parameters["b"]))


def nn_model(X, Y, num_iterations=10, print_cost=False):

    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[1]

    parameters = initialize_parameters(n_x, n_y)

    for i in range(0, num_iterations):

        Y_hat = forward_propagation(X, parameters)

        cost = compute_cost(Y_hat, Y)

        parameters = w3_tools.train_nn(parameters, Y_hat, X, Y)

        if print_cost:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters


parameters = nn_model(X, Y, num_iterations=15, print_cost=True)
print("W = " + str(parameters["W"]))
print("b = " + str(parameters["b"]))

W_simple = parameters["W"]
b_simple = parameters["b"]

X_pred = np.array([-0.95, 0.2, 1.5])

fig, ax = plt.subplots()
plt.scatter(X, Y, color="black")

plt.xlabel("$x$")
plt.ylabel("$y$")

X_line = np.arange(np.min(X[0, :]), np.max(X[0, :]) * 1.1, 0.1)
ax.plot(X_line, W_simple[0, 0] * X_line + b_simple[0, 0], "r")
ax.plot(X_pred, W_simple[0, 0] * X_pred + b_simple[0, 0], "bo")
plt.plot()
plt.show()

#***********************************************************************************************************************

#Dataset

df = pd.read_csv('data/house_prices_train.csv')

X_multi = df[['GrLivArea', 'OverallQual']]
Y_multi = df['SalePrice']

print(f"X_multi:\n{X_multi}\n")
print(f"Y_multi:\n{Y_multi}\n")

#Normalization

X_multi_norm = (X_multi - np.mean(X_multi))/np.std(X_multi)
Y_multi_norm = (Y_multi - np.mean(Y_multi))/np.std(Y_multi)

X_multi_norm = np.array(X_multi_norm).T
Y_multi_norm = np.array(Y_multi_norm).reshape((1, len(Y_multi_norm)))

print ('The shape of X: ' + str(X_multi_norm.shape))
print ('The shape of Y: ' + str(Y_multi_norm.shape))
print ('I have m = %d training examples!' % (X_multi_norm.shape[1]))

X_multi_norm = np.array(X_multi_norm).T
Y_multi_norm = np.array(Y_multi_norm).reshape((1, len(Y_multi_norm)))

print ('The shape of X: ' + str(X_multi_norm.shape))
print ('The shape of Y: ' + str(Y_multi_norm.shape))
print ('I have m = %d training examples!' % (X_multi_norm.shape[1]))
