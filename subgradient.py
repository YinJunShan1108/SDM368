import numpy as np

# Objective function
def f(x):
    return x**2 + np.abs(x)

# Subgradient function
def subgradient(x):
    if x > 0:
        return 2 * x + 1
    elif x < 0:
        return 2 * x - 1
    else:
        return -1

# Subgradient descent method
def subgradient_descent(x0, alpha, tol):
    x = x0
    iter_count = 0
    while np.linalg.norm(x) >= tol:
        grad = subgradient(x)
        x = x - alpha * grad  # Update x
        iter_count += 1
        print(f"Iteration {iter_count}: x = {x}, f(x) = {f(x)}")
    return x

# Initial point
x0 = 1.0
# Constant step size
alpha = 0.001
# Stopping criterion
tolerance = 1e-3

# Execute subgradient descent
optimal_x = subgradient_descent(x0, alpha, tolerance)

print(f"Optimal solution: {optimal_x}, f(x) = {f(optimal_x)}")
