import numpy as np

# 定义目标函数
def f(x):
    return (x[0] ** 2 - x[1]) ** 2 + (x[0] - 1) ** 2

# 计算梯度
def grad_f(x):

    dfdx1 = 4 * x[0] * (x[0] ** 2 - x[1]) + 2 * (x[0] - 1)
    dfdx2 = -2 * (x[0] ** 2 - x[1])
    return np.array([dfdx1, dfdx2])

# 定义步长更新策略
def fixed_step():
    alpha = 0.001
    return alpha

def backtracking_line_search(x, grad):
    alpha, rho, c = 1.0, 0.5, 1e-4
    while f(x - alpha * grad) > f(x) - c * alpha * np.dot(grad, grad):
        alpha *= rho
    return alpha

def decaying_step(k):
    a0 = 0.001
    return a0 / (1 + k)

def sqrt_decay_step(k):
    a0 = 0.001
    return a0 / np.sqrt(1 + k)



# 初始条件
x = np.array([-2, 0])  # 初始点

# 1. fixed_step 2. backtracking_line_search 3. decaying_step 4. sqrt_decay_step
step_size_name = 'backtracking_line_search'

# 停止准则
tolerance = 1e-4

# 迭代过程
iteration = 0
while np.linalg.norm(grad_f(x)) >= tolerance and iteration < 500:
    grad = grad_f(x)

    if step_size_name == 'fixed_step':
        alpha = fixed_step()

    elif step_size_name == 'backtracking_line_search':
        alpha = backtracking_line_search(x, grad)

    elif step_size_name == "decaying_step":
        alpha = decaying_step(iteration)

    elif step_size_name == "sqrt_decay_step":
        alpha = sqrt_decay_step(iteration)

    x = x - alpha * grad  # 更新 x
    iteration += 1

    # 打印每次迭代的信息
    print(f"Iteration {iteration}: x = {x}, f(x) = {f(x)}, ||grad_f(x)|| = {np.linalg.norm(grad)}")

# 输出结果
print("\n")
print(fr"step-size: {step_size_name}")
print("Optimal solution:", x)
print("Objective function value:", f(x))
print("Iterations:", iteration)
