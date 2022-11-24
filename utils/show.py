import numpy as np
from matplotlib import pyplot as plt

import Constants
import lab4


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


def show_borders(x_borders, y_borders, colors_array, labels_array,
                 markers_array):

    for i in range(0, len(x_borders)):
        plt.plot(x_borders[i], y_borders[i], color=colors_array[i], label=labels_array[i], marker=markers_array[i])
    plt.legend()

def show_separating_hyperplanes(title, samples0, samples1, W_array, colors_array, labels_array,
                                markers_array):
    plt.xlim(left=Constants.left, right=Constants.right)
    plt.ylim(bottom=Constants.bot, top=Constants.top)
    plt.title(title)
    plt.plot(samples0[0], samples0[1], 'r.')
    plt.plot(samples1[0], samples1[1], 'b.')

    x_borders = []
    y_borders = []
    x_range = Constants.x_range_lab6
    for i in range(0, len(W_array)):
        W = W_array[i]
        w = W[0]
        wn = W[1]
        x, y = lab4.get_border_lin_classificator(w, wn, x_range)
        x_borders.append(x)
        y_borders.append(y)
        x_borders.append(x + 1 / w[0])
        y_borders.append(y)
        x_borders.append(x - 1 / w[0])
        y_borders.append(y)
        show_borders(np.array(x_borders), np.array(y_borders), colors_array[i], labels_array[i], markers_array[i])
        x_borders = []
        y_borders = []


def show_sup_vectors(sup_vectors0, sup_vectors1):
    if sup_vectors0 is not None and sup_vectors0.size > 0:
        plt.scatter(sup_vectors0[0, :], sup_vectors0[1, :], marker='o', color='c', alpha=0.6)
    if sup_vectors1 is not None and sup_vectors1.size > 0:
        plt.scatter(sup_vectors1[0, :], sup_vectors1[1, :], marker='o', color='fuchsia', alpha=0.6)


def show_bayes_border(border, colol, label, marker):
    plt.scatter(border[:, 0], border[:, 1], marker=marker, color=colol, label=label)
