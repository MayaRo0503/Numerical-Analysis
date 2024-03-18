import numpy as np
from numpy.linalg import norm

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

def is_diagonally_dominant(mat):
    if mat is None:
        return False

    d = np.diag(np.abs(mat))  # Find diagonal coefficients
    s = np.sum(np.abs(mat), axis=1) - d  # Find row sum without diagonal

    return np.all(d >= s)

def jacobi_iterative(A, b, X0, TOL=0.001, N=200):
    n = len(A)
    k = 0

    if is_diagonally_dominant(A):
        print('Matrix is diagonally dominant - performing Jacobi algorithm\n')
        print("Iteration" + "\t\t\t".join([" {:>12}".format(var) for var in ["x{}".format(i) for i in range(1, len(A) + 1)]]))
        print("-" * 100)

        while k < N:
            x = np.zeros(n, dtype=np.double)
            for i in range(n):
                sigma = 0
                for j in range(n):
                    if j != i:
                        sigma += A[i][j] * X0[j]
                x[i] = (b[i] - sigma) / A[i][i]

            print("{:<15} ".format(k + 1) + "\t\t".join(["{:<15} ".format(val) for val in x]))

            if norm(x - X0, np.inf) < TOL:
                return tuple(x)

            k += 1
            X0 = x.copy()

        print("Maximum number of iterations exceeded")
        return tuple(x)

    else:
        A_modified = DominantDiagonalFix(A)
        if A_modified is not None:
            print('Matrix has been modified to create a dominant diagonal - performing Jacobi algorithm\n')
            print("Iteration" + "\t\t\t".join([" {:>12}".format(var) for var in ["x{}".format(i) for i in range(1, len(A_modified) + 1)]]))
            print("-" * 100)

            while k < N:
                x = np.zeros(n, dtype=np.double)
                for i in range(n):
                    sigma = 0
                    for j in range(n):
                        if j != i:
                            sigma += A_modified[i][j] * X0[j]
                    x[i] = (b[i] - sigma) / A_modified[i][i]

                print("{:<15} ".format(k + 1) + "\t\t".join(["{:<15} ".format(val) for val in x]))

                if norm(x - X0, np.inf) < TOL:
                    return tuple(x)

                k += 1
                X0 = x.copy()

            print("Maximum number of iterations exceeded")
            return tuple(x)
        else:
            print("Matrix is not diagonally dominant and cannot be modified to create a dominant diagonal.")
            return None

if __name__ == "__main__":
    A = np.array([[4, 2, 0], [2, 10, 4], [0, 4, 5]])
    b = np.array([2, 6, 5])
    x = np.zeros_like(b, dtype=np.double)

    solution = jacobi_iterative(A, b, x)
    print("\nApproximate solution:", solution)