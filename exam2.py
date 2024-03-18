import numpy as np
from numpy.linalg import norm, inv
import math


def scalar_multiplication_elementary_matrix(n, row_index, scalar):
    if row_index < 0 or row_index >= n:
        raise ValueError("Invalid row index.")

    if scalar == 0:
        raise ValueError("Scalar cannot be zero for row multiplication.")

    elementary_matrix = np.identity(n)
    elementary_matrix[row_index, row_index] = scalar

    return np.array(elementary_matrix)

def norm(mat):
    size = len(mat)
    max_row = 0
    for row in range(size):
        sum_row = 0
        for col in range(size):
            sum_row += abs(mat[row][col])
        if sum_row > max_row:
            max_row = sum_row
    return max_row

def make_diagonal_nonzero(matrix):
    n = len(matrix)

    for k in range(n):
        if matrix[k, k] == 0:
            # Find a non-zero element in the same column below the current zero diagonal element
            for b in range(k + 1, n):
                if matrix[b, k] != 0:
                    # Swap rows to make the diagonal element nonzero
                    matrix[[k, b], :] = matrix[[b, k], :]


    return matrix


def gaussianElimination(mat):
    N = len(mat)

    singular_flag = forward_substitution(mat)

    if singular_flag != -1:

        if mat[singular_flag][N]:
            return "Singular Matrix (Inconsistent System)"
        else:
            return "Singular Matrix (May have infinitely many solutions)"

    # if matrix is non-singular: get solution to system using backward substitution
    return backward_substitution(mat)


# function for elementary operation of swapping two rows
def swap_row(mat, i, j):
    N = len(mat)
    for k in range(N + 1):
        temp = mat[i][k]
        mat[i][k] = mat[j][k]
        mat[j][k] = temp


def forward_substitution(mat):
    N = len(mat)
    for k in range(N):
        pivot_row = k
        v_max = mat[pivot_row][k]
        for i in range(k + 1, N):
            if abs(mat[i][k]) > v_max:
                v_max = mat[i][k]
                pivot_row = i
        if not mat[k][pivot_row]:
            return k  # Matrix is singular

        # Swap the current row with the pivot row
        if pivot_row != k:
            swap_row(mat, k, pivot_row)
        # End Partial Pivoting

        for i in range(k + 1, N):

            #  Compute the multiplier
            m = mat[i][k] / mat[k][k]

            # subtract fth multiple of corresponding kth row element
            for j in range(k + 1, N + 1):
                mat[i][j] -= mat[k][j] * m

            # filling lower triangular matrix with zeros
            mat[i][k] = 0
    # print(mat)
    return -1


# function to calculate the values of the unknowns
def backward_substitution(mat):
    N = len(mat)
    x = np.zeros(N)  # An array to store solution

    # Start calculating from last equation up to the first
    for i in range(N - 1, -1, -1):

        x[i] = mat[i][N]

        # Initialize j to i+1 since matrix is upper triangular
        for j in range(i + 1, N):
            x[i] -= mat[i][j] * x[j]

        x[i] = (x[i] / mat[i][i])

    # print("\n", x)
    return x


def max_steps(a, b, err):
    s = int(np.floor(- np.log2(err / (b - a)) / np.log2(2) - 1))
    return s



def bisection_method(f, a, b, tol=1e-6):
    # if np.sign(a) == np.sign(b):
    #     raise Exception("The scalars a and b do not bound a root")
    c, k = 0, 0
    steps = max_steps(a, b, tol)  # calculate the max steps possible

    #print("{:<10} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15}".format("Iteration", "a", "b", "f(a)", "f(b)", "c", "f(c)"))

    # while the diff af a&b is not smaller than tol, and k is not greater than the max possible steps
    while abs(b - a) > tol and k <= steps:
        c = (a + b) / 2  # Calculation of the middle value

        if f(c) == 0:
            return c  # Procedure completed successfully

        if f(c) * f(a) < 0:  # if sign changed between steps
            b = c  # move forward
        else:
            a = c  # move backward

        #print("{:<10} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f} {:<15.6f}".format(k, a, b, f(a), f(b), c, f(c)))
        k += 1

    return c  # return the current root


def find_all_roots(f, a, b, tol=1e-6):
    roots = []
    x = np.linspace(a, b, 1000)  # Divide the interval into smaller sub-intervals

    for i in range(len(x) - 1):
        if np.sign(f(x[i])) != np.sign(f(x[i + 1])):
            root = bisection_method(f, x[i], x[i + 1], tol)
            roots.append(root)

    return roots

#Date:18.3.23
#Group members: Andrey Romanchuk 323694059;Shahar Dahan 207336355;Maya Rozenberg 313381600
#Source Git: https://github.com/Shmulls/Numerical-analysis/blob/master/Ex2-Mtx.py
#Private Git: https://github.com/AndreyRomanchuk91/SecondYear/tree/master/Numerical%20analysis
#Name: Andrey Romanchuk

if __name__ == '__main__':

    np.set_printoptions(suppress=True, precision=4)
    A_b = [[-1, 1, 3, -3, 1, -1],
           [3, -3, -4, 2, 3, 18],
           [2, 1, -5, -3, 5, 6],
           [-5, -6, 4, 1, 3, 22],
           [3, -2, -2, -3, 5, 10]]


    result = gaussianElimination(A_b)
    if isinstance(result, str):
        print(result)
    else:
        print("\nSolution for the system:")
        for x in result:
            print("{:.6f}".format(x))
    #print("the norm of the matrix plus the question number: ", norm(A_b) + 3)
    f = lambda x: (1 * x ** 5 - 6 * x ** 2 - 1) / (7 * x ** 3 + 1)

    # Adjust the interval to avoid the singularity
    a = -2
    b = 2

    roots = find_all_roots(f, a, b)
    print(f"\nThe equation f(x) has approximate roots at {roots}")
