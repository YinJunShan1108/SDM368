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
def subgradient_descent(x0, tol):
    x = x0
    iteration = 0
    while np.linalg.norm(x) >= tol:
        grad = subgradient(x)
        if step_size_name == 'fixed_step':
            alpha = fixed_step()

        elif step_size_name == "decaying_step":
            alpha = decaying_step(iteration)

        x = x - alpha * grad  # Update x
        iteration += 1
        print(f"Iteration {iteration}: x = {x}, f(x) = {f(x)}")
    return x, iteration

# Step update policy
def fixed_step():
    alpha = 0.001
    return alpha

def decaying_step(k):
    a0 = 1
    return a0 / (1 + k)

# 1. fixed_step 2. decaying_step
step_size_name = 'decaying_step'
# Initial point
x0 = 1.0

# Stopping criterion
tolerance = 1e-3

# Execute subgradient descent
x, iteration = subgradient_descent(x0, tolerance)

# print
print("\n")
print(fr"step-size: {step_size_name}")
print("Optimal solution:", x)
print("Objective function value:", f(x))
print("Iterations:",iteration)