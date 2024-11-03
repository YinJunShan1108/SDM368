import cvxpy as cp
import numpy as np

# 定义变量
x = cp.Variable(2)

# 提取变量
x1 = x[0]
x2 = x[1]

x0 = np.array([-1,3])

# 定义目标函数
objective = cp.Minimize(0.5 * ((x1 - x0[0])**2 + (x2 - x0[1])**2)**2)

# 定义约束条件
constraints = [x1 <= -2, x2 >= 2]

# 构造并求解问题
problem = cp.Problem(objective, constraints)
problem.solve()

# 输出结果
print("最优解 x:", x.value)
print("最优值:", problem.value)
print(x0)