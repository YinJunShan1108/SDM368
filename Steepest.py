import numpy as np


# 定义目标函数
def f(x):
    return 100 * (x[0]**2 - x[1])**2 + (x[0] - 1)**2

# 计算梯度
def grad_f(x):
    dfdx1 = 400 * x[0] * (x[0]**2 - x[1]) + 2 * (x[0] - 1)
    dfdx2 = -200 * (x[0]**2 - x[1])
    return np.array([dfdx1, dfdx2])

