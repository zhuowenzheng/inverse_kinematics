import numpy as np
from rosenbrock import f_objective, gradients, hessian


def BFGS(x, alpha_0, gamma, epsilon, f_obj, grad, iter_max=50, iter_maxLS=20):
    n = len(x)  # Get the dimension of the vector x

    A = np.eye(n)  # Initialize the Hessian approximation matrix to the identity matrix
    x_0 = np.zeros(x.shape)
    g_0 = np.zeros(x.shape)

    for iter in range(iter_max):
        # Evaluate the objective function and its gradient at the current point
        f = f_obj(x)
        g = grad(x)

        if iter > 0:
            s = x - x_0  # Compute the change in x
            y = g - g_0  # Compute the change in the gradient
            rho = 1 / np.dot(y, s)  # Compute the scaling factor rho
            A = (np.eye(n) - rho * np.outer(s, y)) @ A @ (np.eye(n) - rho * np.outer(y, s)) + rho * np.outer(s,
                                                                                                             s)  # Update the Hessian approximation matrix

        p = -A @ g  # Compute the search direction

        # Line search
        a = alpha_0
        delta_x = np.zeros(x.shape)
        for iterLS in range(iter_maxLS):
            delta_x = a * p
            f1 = f_obj(x + delta_x)
            if f1 < f:
                break
            else:
                a *= gamma  # Reduce the step size

        # Update the current point and the previous values of x and g
        x_0 = x.copy()
        g_0 = g.copy()
        x += delta_x  # Take a step along the search direction
        if n == 1:
            while x > np.pi:
                x -= 2 * np.pi
            while x < -np.pi:
                x += 2 * np.pi
        print("iter = ", iter, "f = ", f, "g =", g, "x = ", x)
        if np.linalg.norm(g) < epsilon:
            print("converged at ", iter)
            break

    # Print the final solution to a precision of 6 digits behind the decimal point
    print("x = ", np.round(x, 5))


if __name__ == "__main__":
    epsilon = 1e-6
    alpha = 1
    gamma = 0.8
    init_x = np.array([-1.0, 0.0])
    BFGS(init_x, alpha, gamma, epsilon, f_objective, gradients)
