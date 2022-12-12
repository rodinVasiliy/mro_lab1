import numpy as np
from matplotlib import pyplot as plt

M1 = np.array([0, 0]).reshape(2, 1)
M2 = np.array([1, 1]).reshape(2, 1)
M3 = np.array([-1, 1]).reshape(2, 1)
R1 = np.array(([0.03, 0.0], [0.0, 0.07]))
R2 = np.array(([0.23, 0.01], [0.02, 0.17]))
R3 = np.array(([0.2, 0.1], [0.1, 0.3]))
P1 = 0.5
P2 = 0.5
P3 = 0.5

# task1 это task2, а task2 это task3
M1_lab6_task1 = np.array([0, 0]).reshape(2, 1)
M2_lab6_task1 = np.array([1.5, 1]).reshape(2, 1)

R1_lab6_task1 = np.array(([0.07, 0.0], [0.0, 0.09]))
R2_lab6_task1 = np.array(([0.13, 0.01], [0.02, 0.07]))

R1_lab6_task2 = np.array(([0.07, 0.05], [0.05, 0.09]))
R2_lab6_task2 = np.array(([0.13, 0.05], [0.02, 0.13]))
P1_lab6_task2 = 0.5
P2_lab6_task2 = 0.5

x_range_lab6 = np.arange(-1, 3, 0.1)
left = -1
right = 3
top = 2
bot = -1

k = 0.25

M1_lab5 = np.array([1, -1]).reshape(2, 1)
M2_lab5 = np.array([2, 2]).reshape(2, 1)
R1_lab5 = np.array((
    [0.43, -0.2],
    [-0.2, 0.56]))
R2_lab5 = np.array((
    [0.4, 0.15],
    [0.15, 0.56]))
