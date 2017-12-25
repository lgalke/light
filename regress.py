import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

from utils import visualize

split = False

x, y = np.loadtxt('Modell.csv', unpack=True)
x = x.reshape(-1, 1)

if split:
    x_train, x_test, y_train, y_test = train_test_split(x, y)
else:
    x_train, x_test = x, x
    y_train, y_test = y, y



mlp = MLPRegressor(hidden_layer_sizes=(100),
                   activation='tanh',  # on par with logistic
                   solver='lbfgs',  # superior to both adam, sgd
                   verbose=0)

print("Fitting", mlp)
mlp.fit(x_train, y_train)

pred = mlp.predict(x_test)

for pred, true in zip(pred, y_test):
    print("Pred: {:.2f} ~ {:.2f} true.".format(pred, true))

r2 = mlp.score(x_test, y_test)
print("R^2 score:", r2)

import matplotlib.pyplot as plt

def model(inp):
    return mlp.predict(inp)

visualize(model, x, y, out="out/mlp.png")

