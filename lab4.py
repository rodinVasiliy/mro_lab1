import matplotlib.pyplot as plt
import numpy as np
from skimage.io import show

import Constants
import lab2
from lab2 import get_bayesian_border_for_normal_classes, get_bayesian_border_for_normal_classes_with_same_cor_matrix


def get_fisher_parameters(m0, m1, b0, b1):
    difM = np.reshape(m1, (2, 1)) - np.reshape(m0, (2, 1))
    sumB = 0.5 * (np.array(b0) + np.array(b1))
    W = np.matmul(np.linalg.inv(sumB), difM)
    D0 = np.matmul(np.matmul(np.transpose(W), b0), W)
    D1 = np.matmul(np.matmul(np.transpose(W), b1), W)
    D0 = D0[0, 0]
    D1 = D1[0, 0]
    tmp = np.matmul(np.transpose(difM), np.linalg.inv(sumB))
    tmp = np.matmul(tmp, (D1 * np.reshape(m0, (2, 1)) + D0 * np.reshape(m1, (2, 1))))
    wn = -tmp[0, 0] / (D0 + D1)
    return np.reshape(W, (2,)), wn


def get_msep_parameters(class0, class1):
    size1 = np.shape(class1)
    size0 = np.shape(class0)
    z1Size = ((size1[0] + 1), size1[1])
    z0Size = ((size0[0] + 1), size0[1])
    z1 = np.ones(z1Size)
    z0 = np.ones(z0Size)
    z1[0:size1[0], 0:size1[1]] = class1
    z0[0:size0[0], 0:size0[1]] = class0
    z0 = -1 * z0

    resSize = (3, (size1[1] + size0[1]))
    z = np.ones(resSize)
    z[0:3, 0:z1Size[1]] = z1
    z[0:3, z1Size[1]:resSize[1]] = z0  # size(3, 400)
    # Z в нашем случае это U
    tmp = np.linalg.inv(np.matmul(z, np.transpose(z)))
    # R - вектор требуемых значений выхода
    R = np.ones((resSize[1], 1))
    W = np.matmul(np.matmul(tmp, z), R)
    return np.reshape(W, (3,))


def calc_acrp_parameters(initVectors):
    arrW = []
    W = 0
    size = np.shape(initVectors)
    W = np.ones((size[0] - 1,))
    arrW.append(W)
    cnt = 0
    flagSNG = 0
    for j in range(0, 20):
        for k in range(0, size[1]):
            x = np.reshape(initVectors[0:-1, k], (1, size[0] - 1))
            r = initVectors[-1, k]
            d = np.matmul(arrW[-1], np.transpose(x))
            if ((d[0] < 0) & (r > 0)) | ((d[0] > 0) & (r < 0)):
                sgn = np.sign(r - d)
                if sgn[0] != flagSNG:
                    cnt += 1
                    flagSNG = sgn[0]
                alpha = pow(cnt, -0.999)
                W = arrW[-1] + alpha * x * sgn
                arrW.append(np.reshape(W, (size[0] - 1,)))
    return arrW


def get_border_lin_classificator(W, wn, x):
    # 0 = W0*x + W1*y + wn -> y = -(W0*x + wn)/W1

    if W[1] != 0:
        y = -(W[0] * x + wn) / W[1]
    else:
        x = -(wn / W[0]) * np.ones(len(x))
        y = np.linspace(-100, 100, len(x))
    return x, y


def print_classificator(fig, pos, class0, class1, dfirst, dsecond, nameAnotherBorder, nameBayes, lineFormat, colors):
    fig.add_subplot(pos)
    plt.xlim(-2, 3)
    plt.ylim(-2, 3)
    plt.plot(class0[0], class0[1], 'r+')
    plt.plot(class1[0], class1[1], 'bx')
    plt.plot(dfirst[0], dfirst[1], 'm|')
    for i in range(1, len(dsecond)):
        plt.plot(dsecond[0], dsecond[i], color=colors[i % len(colors)], linestyle=lineFormat)
    plt.legend(["class Red", "class Blue", nameAnotherBorder + " border", nameBayes + " border"])
    return fig


def print_classificator_for_diff_corr_matrix(fig, pos, class0, class1, dfirst, dsecond, nameAnotherBorder, nameBayes,
                                             lineFormat, colors):
    fig.add_subplot(pos)
    plt.xlim(-2, 3)
    plt.ylim(-2, 3)
    plt.plot(class0[0], class0[1], 'r+')
    plt.plot(class1[0], class1[1], 'bx')
    plt.scatter(dfirst[:, 0], dfirst[:, 1], color='m', s=10)
    # c = ['r', 'y', 'g', 'c', 'b', 'm']
    for i in range(1, len(dsecond)):
        plt.plot(dsecond[0], dsecond[i], color=colors[i % len(colors)], linestyle=lineFormat)
    plt.legend(["class Red", "class Blue", nameAnotherBorder + " border", nameBayes + " border"])
    return fig


if __name__ == '__main__':
    # lab 4
    feature1, feature2 = lab2.load_features('C:\\mro_lab1\\two_classes.npy')
    _, feature4, feature5 = lab2.load_features('C:\\mro_lab1\\three_classes.npy')

    size1 = np.shape(feature1)
    size2 = np.shape(feature2)
    size4 = np.shape(feature4)

    M1 = Constants.M1
    M2 = Constants.M2
    M3 = Constants.M3
    R1 = Constants.R1
    R2 = Constants.R2
    R3 = Constants.R3
    P1 = Constants.P1
    P2 = Constants.P2
    P3 = Constants.P3

    # task 4.1
    # Классификатор, максимизирующий критерий Фишера
    W1, wn1 = get_fisher_parameters(M1, M2, R1, R1)
    W2, wn2 = get_fisher_parameters(M1, M2, R1, R2)

    x_array = np.linspace(-2, 3, 100)
    thresh = np.log(P2 / P1)
    bayes_border_1 = get_bayesian_border_for_normal_classes_with_same_cor_matrix(x_array, M1, M2, R1, thresh=thresh)

    dFisher_border_1 = get_border_lin_classificator(W1, wn1, x_array)

    bayes_border_2 = get_bayesian_border_for_normal_classes(x_array, M1, M2, R1, R2, np.log(P2 / P1))
    dFisher_border_2 = get_border_lin_classificator(W2, wn2, x_array)

    fig = plt.figure(figsize=(16, 7))
    fig = print_classificator(fig, 121, feature1, feature2, (x_array, bayes_border_1),
                              dFisher_border_1, "Fisher",
                             "Bayes", "--", ["y", "y"])
    fig = print_classificator_for_diff_corr_matrix(fig, 122, feature1, feature4, bayes_border_2,
                                                   dFisher_border_2, "Fisher",
                                              "Bayes", "--", ["y", "y"])
    show()
    # task 4.2
    # Классификатор, минимизирующий СКО
    Wmse1 = get_msep_parameters(feature1, feature2)
    dMSE1 = get_border_lin_classificator(Wmse1[0:2], Wmse1[-1], x_array)

    Wmse2 = get_msep_parameters(feature1, feature4)
    dMSE2 = get_border_lin_classificator(Wmse2[0:2], Wmse1[-1], x_array)

    fig1 = plt.figure(figsize=(16, 7))
    fig1 = print_classificator(fig1, 121, feature1, feature2, (x_array, bayes_border_1), dMSE1,
                              "MSE", "Bayes", "--", ["y", "y"])
    fig1 = print_classificator_for_diff_corr_matrix(fig1, 122, feature1, feature4, bayes_border_2, dMSE2, "MSE",
                                               "Bayes", "--", ["y", "y"])
    show()

    # task 4.3
    # Классификатор Роббинса-Монро
    Z0 = np.ones((size1[0] + 2, size1[1]))
    Z0[-1] = Z0[-1] * -1
    Z1 = np.ones((size2[0] + 2, size2[1]))
    Z2 = np.ones((size4[0] + 2, size4[1]))

    Z0[0:size1[0], 0:size1[1]] = feature1
    Z1[0:size2[0], 0:size2[1]] = feature2
    Z2[0:size4[0], 0:size4[1]] = feature4

    xz = []
    xy = []
    for i in range(size1[1]):
        xz.append(Z0[:, i])
        xz.append(Z1[:, i])

        xy.append(Z0[:, i])
        xy.append(Z2[:, i])

    xz = np.transpose(xz)
    xy = np.transpose(xy)
    # print(np.shape(xz))
    # Z = np.concatenate((Z0, Z1), axis=1)

    Wrobbins1 = calc_acrp_parameters(xz)
    arrBorders1 = [x_array]
    for w in Wrobbins1:
        tmpY = get_border_lin_classificator(w[0:2], w[-1], x_array)
        arrBorders1.append(tmpY[1])

    c = ["r", "orange", "y", "g", "darkgreen", "c", "b", "m"]
    fig7 = plt.figure(figsize=(16, 7))
    fig7 = print_classificator(fig7, 121, feature1, feature2, (x_array, bayes_border_1),
                              arrBorders1[0:40:4],
                              "Bayes", "Robbins: sample B", "--", c)
    resd1 = [arrBorders1[0], arrBorders1[-1]]
    fig7 = print_classificator(fig7, 122, feature1, feature2, (x_array, bayes_border_1), resd1,
                              "Bayes",
                              "Robbins: sample B", "--", c)
    show()
    Wrobbins2 = calc_acrp_parameters(xy)
    arrBorders2 = [x_array]
    for w in Wrobbins2:
        tmpY = get_border_lin_classificator(w[0:2], w[-1], x_array)
        arrBorders2.append(tmpY[1])

    fig8 = plt.figure(figsize=(16, 7))
    fig8 = print_classificator_for_diff_corr_matrix(fig8, 121, feature1, feature4, bayes_border_2, arrBorders2[0:40:4],
                                               "Bayes", "Robbins: dif B", "--", c)
    resd2 = [arrBorders2[0], arrBorders2[-1]]
    fig8 = print_classificator_for_diff_corr_matrix(fig8, 122, feature1, feature4, bayes_border_2, resd2,
                                               "Bayes", "Robbins: dif B", "--", c)
    show()

    # выбор начального W не влияет на сходимость
    # последовательность: 0.5<b<=1 при большей степени коэфф 1/k^b становится меньше -> сходится плавнее, но медленней
    # и наоборот, при меньшей степени сходится быстрее, но может долго колебаться возле нужной границы
