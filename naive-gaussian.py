def forward_elimination(A, b):
    """
    Performs forward elimination (naïve Gaussian).
    """
    n = len(A)
    
    for k in range(n - 1):  # pivot row
        for i in range(k + 1, n):  # rows below pivot
            if A[k][k] == 0:
                raise ZeroDivisionError("Zero pivot encountered!")
            factor = A[i][k] / A[k][k]
            
            for j in range(k + 1, n):  # update rest of row
                A[i][j] -= factor * A[k][j]
            
            b[i] -= factor * b[k]
            A[i][k] = 0  # eliminate explicitly

    return A, b


# Example usage (just forward elimination)
A = [
    [2, 1, -1],
    [-3, -1, 2],
    [-2, 1, 2]
]
b = [8, -11, -3]

A_mod, b_mod = forward_elimination(A, b)

print("Upper Triangular A:")
for row in A_mod:
    print(row)
print("Modified b:", b_mod)