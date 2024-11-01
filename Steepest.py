import numpy as np

# 目标函数
def f(x):
    return 100 * (x[1] - x[0])**2 - (x[0] - 1)**2

# 目标函数的梯度
def df(x):
    grad = np.zeros(2)
    grad[0] = -400 * (x[1] - x[0]) + 2 * (x[0] - 1)
    grad[1] = 200 * (x[1] - x[0])
    return grad

# 最速下降法（梯度下降法）
def gradient_descent(starting_point, learning_rate, tol, max_iter=10000):
    x = starting_point
    for i in range(max_iter):
        grad = df(x)
        x = x - learning_rate * grad
        if np.linalg.norm(grad, 2) < tol:
            print(f"Converged at iteration {i+1}")
            break
    return x

# 初始点
starting_point = np.array([1.0, 1.0])
# 学习率
learning_rate = 0.2
# 停机准则
tolerance = 1e-4

# 执行梯度下降法
optimal_x = gradient_descent(starting_point, learning_rate, tolerance)
print(f"Optimal x found: {optimal_x}, f(optimal_x) = {f(optimal_x)}")