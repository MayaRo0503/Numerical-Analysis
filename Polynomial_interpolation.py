
def polynomialInterpolation(table_points, x):
    matrix = [[point[0] ** i for i in range(len(table_points))] for point in table_points]  # Makes the initial matrix

    b = [[point[1]] for point in table_points]

    print("The matrix obtained from the points: ", '\n', np.array(matrix))
    print("\nb vector: ", '\n', np.array(b))
    matrixSol = np.linalg.solve(matrix, b)

    result = sum([matrixSol[i][0] * (x ** i) for i in range(len(matrixSol))])
    print("\nThe polynom:")
    print('P(X) = ' + '+'.join(['(' + str(matrixSol[i][0]) + ') * x^' + str(i) + ' ' for i in range(len(matrixSol))]))
    print(f"\nThe Result of P(X={x}) is:")
    print(result)
    return result


if __name__ == '__main__':
    table_points = [(0, 0), (1, 0.8415), (2, 0.9093), (3, 0.1411), (4, -0.7568), (5, -0.9589), (6, -0.2794)]
    x = 1.28
    print("----------------- Interpolation & Extrapolation Methods -----------------\n")
    print("Table Points: ", table_points)
    print("Finding an approximation to the point: ", x, '\n')
    polynomialInterpolation(table_points, x)
    print("\n---------------------------------------------------------------------------\n")
