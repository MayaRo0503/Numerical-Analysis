
def linearInterpolation(table_points, point):
    p = []
    result = 0
    flag = 1
    for i in range(len(table_points)):
        p.append(table_points[i][0])
    for i in range(len(p) - 1):
        if p[i] <= point <= p[i + 1]:
            x1 = table_points[i][0]
            x2 = table_points[i + 1][0]
            y1 = table_points[i][1]
            y2 = table_points[i + 1][1]
            result = (((point - x2) * y1) / (x1 - x2) + (((point - x1) * y2) / (x2 - x1)))
            print("\nThe approximation (interpolation) of the point ", point, " is: ", round(result, 4))
            flag = 0
    if flag:
        if point < table_points[0][0]:
            x1 = table_points[0][0]
            x2 = table_points[1][0]
            y1 = table_points[0][1]
            y2 = table_points[1][1]
            m = (y1 - y2) / (x1 - x2)
            result = y1 + (m * (point - x1))
            print("\nThe approximation (extrapolation) of the point ", point, " is: ", round(result, 4))
        elif point > table_points[-1][0]:
            x1 = table_points[-1][0]
            x2 = table_points[-2][0]
            y1 = table_points[-1][1]
            y2 = table_points[-2][1]
            m = (y1 - y2) / (x1 - x2)
            result = y1 + (m * (point - x1))
            print("\nThe approximation (extrapolation) of the point ", point, " is: ",
                  round(result, 4))
        else:
            print("Point not found in the table range. Unable to interpolate or extrapolate.")



if __name__ == '__main__':

    table_points = [(1, 0.8415), (2, 0.9093), (3, 0.1411), (4, -0.7568), (5, -0.9589), (6, -0.2794)]
    x = 7.28
    print("----------------- Interpolation & Extrapolation Methods -----------------\n")
    print("Table Points: ", table_points)
    print("Finding an approximation to the point: ", x)
    linearInterpolation(table_points, x)
    print("\n---------------------------------------------------------------------------\n")

