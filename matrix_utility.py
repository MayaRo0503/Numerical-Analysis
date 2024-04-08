import numpy as np

def print_matrix(matrix):
    for row in matrix:
        for element in row:
            print(element, end=" ")
        print()
    print()

def is_diagonally_dominant(mat):
    if mat is None:
        return False

    d = np.diag(np.abs(mat))
    s = np.sum(np.abs(mat), axis=1) - d
    return (d > s).all()


def is_square_matrix(mat):
    if mat is None:
        return False

    rows = len(mat)
    for row in mat:
        if len(row) != rows:
            return False
    return True


def reorder_dominant_diagonal(matrix):
    n = len(matrix)
    permutation = np.argsort(np.diag(matrix))[::-1]
    reordered_matrix = matrix[permutation][:, permutation]
    return reordered_matrix


def DominantDiagonalFix(matrix):
    """
    Function to change a matrix to create a dominant diagonal
    :param matrix: Matrix nxn
    :return: Change the matrix to a dominant diagonal
    """
    #Check if we have a dominant for each column
    dom = [0]*len(matrix)
    result = list()
   # Find the largest organ in a row
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if (matrix[i][j] > sum(map(abs,map(int,matrix[i])))-matrix[i][j]) :
                dom[i]=j
    for i in range(len(matrix)):
        result.append([])
        # Cannot dominant diagonal
        if i not in dom:
            print("Couldn't find dominant diagonal.")
            return matrix
    # Change the matrix to a dominant diagonal
    for i,j in enumerate(dom):
        result[j]=(matrix[i])
    return result


def swap_rows_elementary_matrix(n, row1, row2):
    elementary_matrix = np.identity(n)
    elementary_matrix[[row1, row2]] = elementary_matrix[[row2, row1]]

    return np.array(elementary_matrix)


def matrix_multiply(A, B):
    if len(A[0]) != len(B):
        raise ValueError("Matrix dimensions are incompatible for multiplication.")

    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]

    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += (A[i][k] * B[k][j])

    return np.array(result)

def row_addition_elementary_matrix(n, target_row, source_row, scalar=1.0):

    if target_row < 0 or source_row < 0 or target_row >= n or source_row >= n:
        raise ValueError("Invalid row indices.")

    if target_row == source_row:
        raise ValueError("Source and target rows cannot be the same.")

    elementary_matrix = np.identity(n)
    elementary_matrix[target_row, source_row] = scalar

    return np.array(elementary_matrix)


def scalar_multiplication_elementary_matrix(n, row_index, scalar):

    if row_index < 0 or row_index >= n:
        raise ValueError("Invalid row index.")

    if scalar == 0:
        raise ValueError("Scalar cannot be zero for row multiplication.")

    elementary_matrix = np.identity(n)
    elementary_matrix[row_index, row_index] = scalar

    return np.array(elementary_matrix)


def partial_pivoting(A,i,N):
    pivot_row = i
    v_max = A[pivot_row][i]
    for j in range(i + 1, N):
        if abs(A[j][i]) > abs(v_max):
            """1 change"""
            v_max = A[j][i]
            pivot_row = j

    if not A[pivot_row][i]:
        return "Singular Matrix"

    if pivot_row != i:
        swap_row(A, i, pivot_row)
    return A


def swap_row(mat, i, j):
    N = len(mat)
    for k in range(N + 1):
        temp = mat[i][k]
        mat[i][k] = mat[j][k]
        mat[j][k] = temp
def backward_substitution(mat):
    #N = len(mat)
    N, M= np.array(mat).shape
    x = np.zeros(N)


    for i in range(N - 1, -1, -1):

        x[i] = mat[i][M-1]

        for j in range(i + 1, N):
            x[i] -= mat[i][j] * x[j]

        x[i] = (x[i] / mat[i][i])
    return x
