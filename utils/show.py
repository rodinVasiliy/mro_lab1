import numpy as np
from matplotlib import pyplot as plt

import lab1

def show(title: str, dataset0: np.array, dataset1: np.array, border_x_arr, border_y_arr, colors, labels):
    plt.figure()
    plt.title(title)

    plt.plot(dataset0[0], dataset0[1], color='red', marker='.')
    plt.plot(dataset1[0], dataset1[1], color='green', marker='+')

    for i in range(len(border_x_arr)):
        plt.plot(border_x_arr[i], border_y_arr[i], color=colors[i], label=labels[i])

    plt.legend()

    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=-4, top=4)