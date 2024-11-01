import numpy as np

def f(x):
    return np.sum((x-1)**2) + np.sum(np.abs(x))

def subgradient(x):
    return 2*(x-1) + np.sign(x)

def prox_operator(v, alpha):
    return np.sign(v) * np.maximum(np.abs(v) - alpha, 0)

def subgradient_method(x0, max_iter=1000, alpha=0.1):
    x = x0
    for k in range(max_iter):
        g = subgradient(x)
        x = prox_operator(x - alpha * g, alpha)
        if np.linalg.norm(g) < 1e-4:
            break
    return x

# 初始点
x0 = np.array([2, 2, 2])
# 运行次梯度方法
x_opt = subgradient_method(x0)
print("Optimal solution:", x_opt)
print("Function value at optimal solution:", f(x_opt))

