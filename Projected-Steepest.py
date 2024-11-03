import cvxpy as cp
import numpy as np

# Define the objective function
def f(x):
    return 100 * (x[0]**2 - x[1])**2 + (x[0] - 1)**2

# gradient
def grad_f(x):
    dfdx1 = 400 * x[0] * (x[0]**2 - x[1]) + 2 * (x[0] - 1)
    dfdx2 = -200 * (x[0]**2 - x[1])
    return np.array([dfdx1, dfdx2])

def project(x):
    x[0] = min(x[0],-2)
    x[1] = max(x[1], 2)

    return x

def project2(x0):
    # Define variables
    x = cp.Variable(2)

    # Extract variables
    x1, x2 = x[0], x[1]

    # Define the objective function
    objective = cp.Minimize(0.5 * ((x1 - x0[0]) ** 2 + (x2 - x0[1]) ** 2) ** 2)


    # Define the constraints
    constraints = [x1 <= -2, x2 >= 2]

    # Construct and solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    return np.round(x.value, decimals=2)


x = np.array([-3,3])
alpha, tolerance = 0.0001, 1e-4

iteration = 0
while True:
    x_old = x.copy()
    x = project(x - alpha * grad_f(x))

    iteration += 1

    print(
        f"Iteration {iteration}: x = {x}, f(x) = {f(x)}, ||x^(k+1) - x^(k)|| / alpha = {np.linalg.norm(x - x_old) / alpha}")
    if np.linalg.norm(x - x_old) / alpha < tolerance:
        break

# 输出结果
print("\nOptimal solution:", x)
print("Objective function value:", f(x))
print("Iterations:", iteration)

