#Date:8.4.2024
#Group members: Andrey Romanchuk 323694059;Shahar Dahan 207336355;Maya Rozenberg 313381600
#Source Git: https://github.com/lihiSabag/Numerical-Analysis-2023.git
#Private Git: https://github.com/MayaRo0503/Numerical-Analysis/tree/main
#Name: Maya Rozenberg
import numpy as np
from matrix_utility import *

def polynomialInterpolation(table_points, x):
    matrix = [[point[0] ** i for i in range(len(table_points))] for point in table_points]  # Makes the initial matrix
    b = [[point[1]] for point in table_points]
    matrixSol = np.linalg.solve(matrix, b)

    result = sum([matrixSol[i][0] * (x ** i) for i in range(len(matrixSol))])
    print(f"\nThe Result of P(X={x}) is:")
    print(result)
    return result

def romberg_integration(func, a, b, n):
    counter = 1
    """
    Romberg Integration

    Parameters:
    func (function): The function to be integrated.
    a (float): The lower limit of integration.
    b (float): The upper limit of integration.
    n (int): The number of iterations (higher value leads to better accuracy).

    Returns:
    float: The approximate definite integral of the function over [a, b].
    """
    h = b - a
    R = np.zeros((20, 20), dtype=float)

    R[0, 0] = 0.5 * h * (func(a) + func(b))

    for i in range(1, 20):
        h /= 2
        sum_term = 0
        counter += 1
        for k in range(1, 2 ** i, 2):
            sum_term += func(a + k * h)

        R[i, 0] = 0.5 * R[i - 1, 0] + h * sum_term

        for j in range(1, i + 1):
            R[i, j] = R[i, j - 1] + (R[i, j - 1] - R[i - 1, j - 1]) / ((4 ** j) - 1)

        if np.round(R[i - 1, i - 1], n) == np.round(R[i, i], n):
            print(counter)
            return R[i, i]

    print(f"we have reach maximum iterations {n}")
    return R[n - 1, n - 1]


def f(x):
    return (2*x**2 + np.cos(2*(np.e)**(-2*x))) / (2*x**3 + x**2 -6)

if __name__ == '__main__':
    table_points = [(1.2,1.2), (1.3,2.3), (1.4,-0.5), (1.5,-0.89), (1.6,-1.37)]
    x_a = 1.25
    x_b = 1.55
    polynomialInterpolation(table_points, x_a)
    polynomialInterpolation(table_points, x_b)
    print("\n---------------------------------------------------------------------------\n")

    b = polynomialInterpolation(table_points, x_a)
    a = polynomialInterpolation(table_points, x_b)
    n = 5
    integral = romberg_integration(f, a, b, n)

    print(f"Approximate integral in range [{a},{b}] is {integral}")
