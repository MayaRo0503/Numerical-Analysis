from colors import bcolors
import numpy as np

"""
Function that find the inverse of non-singular matrix
The function performs elementary row operations to transform it into the identity matrix. 
The resulting identity matrix will be the inverse of the input matrix if it is non-singular.
 If the input matrix is singular (i.e., its diagonal elements become zero during row operations), it raises an error.
"""
def row_addition_elementary_matrix(size, target_row, source_row, scalar):
    # Create an identity matrix
    elementary_matrix = np.identity(size)
    # Modify the corresponding elements to achieve row addition
    elementary_matrix[target_row, source_row] = scalar
    return elementary_matrix

def scalar_multiplication_elementary_matrix(size, row, scalar):
    # Create an elementary matrix with the specified scalar multiplication
    elementary_matrix = np.identity(size)
    elementary_matrix[row, row] = scalar
    return elementary_matrix

def solve_linear_system(coeff_matrix, b_vector):
    try:
        x_vector = np.linalg.solve(coeff_matrix, b_vector)
        return x_vector
    except np.linalg.LinAlgError:
        raise ValueError("System of equations is singular or not square, cannot be solved.")


def matrix_inverse(matrix):
    print(bcolors.OKBLUE, f"=================== Finding the inverse of a non-singular matrix using elementary row operations ===================\n {matrix}\n", bcolors.ENDC)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")

    n = matrix.shape[0]
    identity = np.identity(n)

    # Perform row operations to transform the input matrix into the identity matrix
    for i in range(n):
        if matrix[i, i] == 0:
            raise ValueError("Matrix is singular, cannot find its inverse.")

        if matrix[i, i] != 1:
            # Scale the current row to make the diagonal element 1
            scalar = 1.0 / matrix[i, i]
            elementary_matrix = scalar_multiplication_elementary_matrix(n, i, scalar)
            print(f"elementary matrix to make the diagonal element 1 :\n {elementary_matrix} \n")
            matrix = np.matmul(elementary_matrix, matrix)
            print(f"The matrix after elementary operation :\n {matrix}")
            print(bcolors.OKGREEN, "------------------------------------------------------------------------------------------------------------------",  bcolors.ENDC)
            identity = np.matmul(elementary_matrix, identity)

        # Zero out the elements above and below the diagonal
        for j in range(n):
            if i != j:
                scalar = -matrix[j, i]
                elementary_matrix = row_addition_elementary_matrix(n, j, i, scalar)
                print(f"elementary matrix for R{j+1} = R{j+1} + ({scalar}R{i+1}):\n {elementary_matrix} \n")
                matrix = np.matmul(elementary_matrix, matrix)
                print(f"The matrix after elementary operation :\n {matrix}")
                print(bcolors.OKGREEN, "------------------------------------------------------------------------------------------------------------------",
                      bcolors.ENDC)
                identity = np.matmul(elementary_matrix, identity)

    return identity


if __name__ == '__main__':

    A = np.array([[1, 0, 2],
                  [2, -1, 3],
                  [4, 1, 8]])

    b = np.array([1, 2, 3])

    try:
        solution = solve_linear_system(A, b)
        A_inverse = matrix_inverse(A)
        print(bcolors.OKBLUE, "\nSolution to the linear system: \n", solution, bcolors.ENDC)
        print(bcolors.OKBLUE, "\nInverse of matrix A: \n", A_inverse)
        print("=====================================================================================================================", bcolors.ENDC)

    except ValueError as e:
        print(str(e))
