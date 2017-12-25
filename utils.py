""" Utility functions such as model visualization """
import matplotlib.pyplot as plt
import numpy as np


def visualize(model,
              samples,
              targets,
              step=.1,
              out=None):
    """
    Visualize (x,y) paired data and model predictions on the range
    from min(x) to max(x). Model is assumed to be a function
    mapping from x to the predictions.
    """
    plt.figure(1)
    plt.scatter(samples, targets, c='g')
    full_x = np.arange(samples.min(), targets.min(), step)
    plt.plot(full_x, model(full_x))
    if not out:
        plt.show()
    else:
        print("Writing plot to file", out)
        plt.savefig(out)
