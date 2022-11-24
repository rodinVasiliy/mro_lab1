import numpy as np
from matplotlib import pyplot as plt

import Constants


def show(title: str, dataset0: np.array, dataset1: np.array, border_x_arr, border_y_arr, colors, labels):
    plt.figure()
    plt.title(title)

    plt.plot(dataset0[0], dataset0[1], 'r.')
    plt.plot(dataset1[0], dataset1[1], 'b.')

    for i in range(len(border_x_arr)):
        plt.plot(border_x_arr[i], border_y_arr[i], color=colors[i], label=labels[i])

    plt.legend()

    plt.xlim(left=Constants.left, right=Constants.right)
    plt.ylim(bottom=Constants.bot, top=Constants.top)
