import numpy as np

# Define the objective function
def f(x):
    return 100 * (x[0]**2 - x[1])**2 + (x[0] - 1)**2

# gradient
def grad_f(x):
    dfdx1 = 400 * x[0] * (x[0]**2 - x[1]) + 2 * (x[0] - 1)
    dfdx2 = -200 * (x[0]**2 - x[1])
    return np.array([dfdx1, dfdx2])

# Step update policy
def fixed_step():
    alpha = 0.001
    return alpha

def backtracking_line_search(x, grad):
    alpha, beta, c = 1.0, 0.1, 1e-4
    while f(x - alpha * grad) > f(x) - c * alpha * np.dot(grad, grad):
        alpha *= beta
    return alpha

def decaying_step(k):
    a0 = 0.001
    return a0 / (1 + k)

def sqrt_decay_step(k):
    a0 = 0.001
    return a0 / np.sqrt(1 + k)



# initial conditions
x = np.array([2, 1])  # initial point

# 1. fixed_step 2. backtracking_line_search 3. decaying_step 4. sqrt_decay_step
step_size_name = 'fixed_step'

# 停stopping criterion
tolerance = 1e-4

# iterative process
iteration = 0
while True:

    if step_size_name == 'fixed_step':
        alpha = fixed_step()

    elif step_size_name == 'backtracking_line_search':
        alpha = backtracking_line_search(x, grad_f(x))

    elif step_size_name == "decaying_step":
        alpha = decaying_step(iteration)

    elif step_size_name == "sqrt_decay_step":
        alpha = sqrt_decay_step(iteration)

    x = x - alpha * grad_f(x)  # Update x
    iteration += 1

    # Print information for each iteration
    print(f"Iteration {iteration}: x = {x}, f(x) = {f(x)}, ||grad_f(x)|| = {np.linalg.norm(grad_f(x))}")
    if np.linalg.norm(grad_f(x)) < tolerance:
        break


# print
print("\n")
print(fr"step-size: {step_size_name}")
print("Optimal solution:", x)
print("Objective function value:", f(x))
print("Iterations:", iteration)
