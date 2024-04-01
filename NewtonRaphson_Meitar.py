#Date:4.4.2024
#Group members: Andrey Romanchuk ID;Shahar Dahan ID;Maya Rozenberg ID:313381600
#Private Git: https://github.com/MayaRo0503/Numerical-Analysis/tree/main
#Name: Maya Rozenberg

import numpy as np


# Function to evaluate the polynomial at a given point
def equation_evaluation(equation, x):
    return equation(x)


# Derivative of the equation
def equation_derivative(x):
    return 12 * x ** 2 - 48


# Newton-Raphson method
def newton_raphson(equation, range_start, range_end, epsilon=0.0001):
    iterations = 0

    # Check for sign change between the endpoints of the range
    if equation_evaluation(equation, range_start) * equation_evaluation(equation, range_end) >= 0:
        print("No roots found in the given range.")
        return None

    # Newton-Raphson iteration loop
    while True:
        x_n = range_end - equation_evaluation(equation, range_end) / equation_evaluation(equation_derivative, range_end)

        # Check termination condition
        if abs(x_n - range_end) < epsilon:
            break

        range_end = x_n
        iterations += 1

    print(f"Root found at x = {x_n} with {iterations} iterations.")
    return x_n, iterations


# Meitar method
def meitar(equation, range_start, range_end, epsilon=0.0001):
    iterations = 0

    # Check for sign change between the endpoints of the range
    if equation_evaluation(equation, range_start) * equation_evaluation(equation, range_end) >= 0:
        print("No roots found in the given range.")
        return None

    # Meitar iteration loop
    while True:
        x_n = (range_start + range_end) / 2

        if abs(range_end - range_start) < epsilon:
            break

        if equation_evaluation(equation, range_start) * equation_evaluation(equation, x_n) < 0:
            range_end = x_n
        else:
            range_start = x_n

        iterations += 1

    print(f"Root found at x = {x_n} with {iterations} iterations.")
    return x_n, iterations


# Main program
if __name__ == "__main__":
    # General parameters
    def my_equation(x):
        return 4 * x ** 3 - 48 * x + 5


    range_start = 3
    range_end = 4  # Here we specify the search range for the root
    epsilon = 0.001  # The accuracy we want to achieve

    print("Running Newton-Raphson method:")
    root, iterations = newton_raphson(my_equation, range_start, range_end, epsilon)
    print(f"Number of iterations: {iterations}")

    print("\nRunning Meitar method:")
    root, iterations = meitar(my_equation, range_start, range_end, epsilon)
    print(f"Number of iterations: {iterations}")
