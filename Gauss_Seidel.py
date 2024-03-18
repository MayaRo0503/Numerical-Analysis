import numpy as np
from numpy.linalg import norm


def is_diagonally_dominant(mat):
    if mat is None:
        return False

    d = np.diag(np.abs(mat))  # Find diagonal coefficients
    s = np.sum(np.abs(mat), axis=1) - d  # Find row sum without diagonal
    return np.all(d >= s)

def DominantDiagonalFix(matrix):
    """
    Function to change a matrix to create a dominant diagonal
    :param matrix: Matrix nxn
    :return: Changed matrix with dominant diagonal, or None if not possible
    """
    n = len(matrix)
    result = [[0] * n for _ in range(n)]  # Initialize result matrix with zeros

    # Find the row with the largest element in each column
    max_rows = [0] * n
    for j in range(n):
        max_val = abs(matrix[0][j])
        max_row = 0
        for i in range(1, n):
            if abs(matrix[i][j]) >= max_val:
                max_val = abs(matrix[i][j])
                max_row = i
        max_rows[j] = max_row

    # Check if a dominant diagonal can be created
    for i in range(n):
        diagonal_element = abs(matrix[max_rows[i]][i])
        row_sum = sum(abs(matrix[max_rows[i]][j]) for j in range(n) if j != i)
        if diagonal_element <= row_sum:
            print("Couldn't find dominant diagonal.")
            return None

    # Create the dominant diagonal matrix
    for i in range(n):
        result[i] = matrix[max_rows[i]]

    return result

def gauss_seidel(A, b, X0, TOL=0.001, N=200):
    n = len(A)
    k = 1

    if is_diagonally_dominant(A):
        print('Matrix is diagonally dominant - preforming gauss seidel algorithm\n')

    else:
        DominantDiagonalFix(A)

    print("Iteration" + "\t\t\t".join([" {:>12}".format(var) for var in ["x{}".format(i) for i in range(1, len(A) + 1)]]))
    print("-----------------------------------------------------------------------------------------------")
    x = np.zeros(n, dtype=np.double)
    while k <= N:

        for i in range(n):
            sigma = 0
            for j in range(n):
                if j != i:
                    sigma += A[i][j] * x[j]
            x[i] = (b[i] - sigma) / A[i][i]

        print("{:<15} ".format(k) + "\t\t".join(["{:<15} ".format(val) for val in x]))

        if norm(x - X0, np.inf) < TOL:
            return tuple(x)

        k += 1
        X0 = x.copy()

    print("Maximum number of iterations exceeded")
    return tuple(x)


if __name__ == '__main__':

    A = np.array([[2, 3, 1], [1, 1, 2], [3, 1, 2]])
    b = np.array([7, 5, 8])
    X0 = np.zeros_like(b)

    solution = gauss_seidel(A, b, X0)
    print("\nApproximate solution:", solution)
