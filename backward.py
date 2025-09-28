def back_substitution(A, b):
    """
    Performs back substitution for an upper triangular matrix.
    """
    n = len(A)
    x = [0] * n

    print("Upper Triangular Matrix A:")
    for row in A:
        print(row)
    print("Modified b:", b)
    print()

    x[n-1] = b[n-1] / A[n-1][n-1]  # last variable

    for i in range(n-2, -1, -1):
        summation = b[i]
        for j in range(i+1, n):
            summation -= A[i][j] * x[j]
        x[i] = summation / A[i][i]

    return x


# Example usage
A = [
    [2, -1, 1],
    [0, 3.5, -2.5],
    [0, 0, -7]
]
b = [3, 1.5, -7]

x = back_substitution(A, b)

print("Solution vector x:", x)