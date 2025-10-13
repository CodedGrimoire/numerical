# --- JACOBI METHOD (pure Python) with edge-case handling ---
import math
import matplotlib.pyplot as plt

EPS_PIVOT = 1e-14

def det_via_elimination(Ain):
    """Return determinant estimate using Gaussian elimination with partial pivoting."""
    A = [row[:] for row in Ain]
    n = len(A)
    det = 1.0
    swap_count = 0
    for i in range(n):
        # partial pivot
        piv = i
        maxabs = abs(A[i][i])
        for r in range(i+1, n):
            if abs(A[r][i]) > maxabs:
                maxabs = abs(A[r][i])
                piv = r
        if maxabs < EPS_PIVOT:
            return 0.0  # singular
        if piv != i:
            A[i], A[piv] = A[piv], A[i]
            swap_count ^= 1  # flip parity
        pivot = A[i][i]
        for r in range(i+1, n):
            m = A[r][i] / pivot
            for c in range(i, n):
                A[r][c] -= m * A[i][c]
    for i in range(n):
        det *= A[i][i]
    if swap_count:
        det = -det
    return det

def is_strictly_diag_dominant(A):
    n = len(A)
    for i in range(n):
        diag = abs(A[i][i])
        off = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diag <= off:
            return False
    return True

def try_fix_zero_diagonal(A, b):
    """Try to swap rows to avoid zero/near-zero diagonal elements. Returns True if fixed/ok."""
    n = len(A)
    for i in range(n):
        if abs(A[i][i]) < EPS_PIVOT:
            # find a row j>i with nonzero in column i
            swap_with = None
            for j in range(i+1, n):
                if abs(A[j][i]) >= EPS_PIVOT:
                    swap_with = j
                    break
            if swap_with is None:
                return False
            A[i], A[swap_with] = A[swap_with], A[i]
            b[i], b[swap_with] = b[swap_with], b[i]
    return True

def jacobi_method(A_in, b_in, tol=1e-3, max_iter=100, diverge_patience=5):
    """
    Jacobi iterative solver with:
      - singularity check (det ≈ 0)
      - zero-diagonal row-swap fix
      - diagonal dominance warning
      - divergence/oscillation early-stop
    Starts from x=0. Returns x or None.
    """
    # deep copies (we may swap rows)
    A = [row[:] for row in A_in]
    b = b_in[:]
    n = len(b)

    # basic shape checks
    if any(len(row) != n for row in A):
        print("Error: A must be square and match b.")
        return None

    # singularity check
    d = det_via_elimination(A)
    if abs(d) < EPS_PIVOT:
        print("Jacobi aborted: det(A) ≈ 0 (singular or ill-conditioned).")
        return None

    # zero diagonal fix
    if not try_fix_zero_diagonal(A, b):
        print("Jacobi aborted: zero/near-zero diagonal could not be fixed by row swap.")
        return None

    # dominance warning
    if not is_strictly_diag_dominant(A):
        print("Warning: A is not strictly diagonally dominant; Jacobi may diverge or be slow.")

    x_old = [0.0]*n
    es_series = []
    worse_streak = 0

    # header
    header = f"{'Iter':<5}" + "".join([f"{('x'+str(i+1)):>12}" for i in range(n)]) + f"{'es(%)':>12}"
    print(header)

    for it in range(1, max_iter+1):
        x_new = [0.0]*n
        for i in range(n):
            s = 0.0
            for j in range(n):
                if j != i:
                    s += A[i][j] * x_old[j]
            x_new[i] = (b[i] - s) / A[i][i]

        # error (∞-norm)
        max_diff = max(abs(x_new[i] - x_old[i]) for i in range(n))
        denom = max(max(abs(v) for v in x_new), 1e-12)
        es = (max_diff / denom) * 100.0
        es_series.append(es)

        print(f"{it:<5}" + "".join([f"{x_new[i]:12.10f}" for i in range(n)]) + f"{es:12.8f}")

        # divergence detector (non-decreasing error streak)
        if it >= 2 and es >= es_series[-2] - 1e-12:
            worse_streak += 1
        else:
            worse_streak = 0
        if worse_streak >= diverge_patience:
            print("\nJacobi stopping early: error not improving (likely divergence/oscillation).")
            break

        x_old = x_new

        if es <= tol:
            print("\nConverged successfully (Jacobi).")
            break

    # plot
    if es_series:
        plt.figure()
        plt.plot(range(1, len(es_series)+1), es_series, marker='o')
        plt.xlabel("Iteration")
        plt.ylabel("Approximate relative error (%)")
        plt.title("Jacobi: es(%) vs Iteration")
        plt.grid(True)
        plt.show()

    return x_old

# ----- example -----
A = [
    [10.0, -1.0,  2.0],
    [-1.0, 11.0, -1.0],
    [ 2.0, -1.0, 10.0]
]
b = [6.0, 25.0, -11.0]

print("=== Jacobi Method (with checks) ===")
xj = jacobi_method(A, b, tol=1e-3, max_iter=100)
print("Final x (Jacobi):", xj)
