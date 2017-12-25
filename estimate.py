""" Estimate parameters a, b, C of y = a * (x/C) ^-b """
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from utils import visualize
rng = np.random

train_X, train_Y = np.loadtxt("Modell.csv", unpack=True)

assert train_X.shape[0] == train_Y.shape[0]
n_samples = train_X.shape[0]


training_epochs = 1000
display_step = 10
learning_rate = 5e-7
momentum = 0.9
use_nesterov = True

X = tf.placeholder("float")
Y = tf.placeholder("float")

a = tf.Variable(rng.randn(), name="a")
b = tf.Variable(1.0 + rng.randn(), name="b")
C = tf.Variable(100.0, name="C")


pred = a * (X / C) ** (-b)

cost = tf.reduce_mean(tf.pow(pred-Y, 2)) / 2

if momentum:
    optimizer = tf.train.MomentumOptimizer(learning_rate,
                                           momentum=momentum,
                                           use_nesterov=use_nesterov).minimize(cost)
else:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            # stochastic
            sess.run(optimizer, feed_dict={X: x, Y: y})
        if (epoch+1) % display_step == 0:
            loss = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(loss), \
                "a=", sess.run(a), "b=", sess.run(b), "C=", sess.run(C))


    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "a=", sess.run(a), "b=", sess.run(b), "C=", sess.run(C), '\n')

    def model(inp):
        return sess.run(a) * (inp / sess.run(C)) ** -sess.run(b)

    visualize(model, train_X, train_Y, out="out/math.png")
