# --- GAUSS–SEIDEL METHOD (pure Python) with edge-case handling ---
import math
import matplotlib.pyplot as plt

EPS_PIVOT = 1e-14

def det_via_elimination(Ain):
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
            return 0.0
        if piv != i:
            A[i], A[piv] = A[piv], A[i]
            swap_count ^= 1
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
    n = len(A)
    for i in range(n):
        if abs(A[i][i]) < EPS_PIVOT:
            sw = None
            for j in range(i+1, n):
                if abs(A[j][i]) >= EPS_PIVOT:
                    sw = j
                    break
            if sw is None:
                return False
            A[i], A[sw] = A[sw], A[i]
            b[i], b[sw] = b[sw], b[i]
    return True

def gauss_seidel_method(A_in, b_in, tol=1e-3, max_iter=100, diverge_patience=5):
    """
    Gauss–Seidel with:
      - singularity check (det ≈ 0)
      - zero-diagonal row-swap fix
      - diagonal dominance warning
      - divergence/oscillation early-stop
    Starts from x=0. Returns x or None.
    """
    A = [row[:] for row in A_in]
    b = b_in[:]
    n = len(b)

    if any(len(row) != n for row in A):
        print("Error: A must be square and match b.")
        return None

    d = det_via_elimination(A)
    if abs(d) < EPS_PIVOT:
        print("Gauss–Seidel aborted: det(A) ≈ 0 (singular or ill-conditioned).")
        return None

    if not try_fix_zero_diagonal(A, b):
        print("Gauss–Seidel aborted: zero/near-zero diagonal could not be fixed by row swap.")
        return None

    if not is_strictly_diag_dominant(A):
        print("Warning: A is not strictly diagonally dominant; Gauss–Seidel may diverge or be slow.")

    x = [0.0]*n
    es_series = []
    worse_streak = 0

    header = f"{'Iter':<5}" + "".join([f"{('x'+str(i+1)):>12}" for i in range(n)]) + f"{'es(%)':>12}"
    print(header)

    for it in range(1, max_iter+1):
        x_prev = x[:]
        for i in range(n):
            s1 = sum(A[i][j]*x[j]     for j in range(0, i))
            s2 = sum(A[i][j]*x_prev[j] for j in range(i+1, n))
            x[i] = (b[i] - s1 - s2) / A[i][i]

        max_diff = max(abs(x[i] - x_prev[i]) for i in range(n))
        denom = max(max(abs(v) for v in x), 1e-12)
        es = (max_diff / denom) * 100.0
        es_series.append(es)

        print(f"{it:<5}" + "".join([f"{x[i]:12.10f}" for i in range(n)]) + f"{es:12.8f}")

        # divergence detector
        if it >= 2 and es >= es_series[-2] - 1e-12:
            worse_streak += 1
        else:
            worse_streak = 0
        if worse_streak >= diverge_patience:
            print("\nGauss–Seidel stopping early: error not improving (likely divergence/oscillation).")
            break

        if es <= tol:
            print("\nConverged successfully (Gauss–Seidel).")
            break

    if es_series:
        plt.figure()
        plt.plot(range(1, len(es_series)+1), es_series, marker='o')
        plt.xlabel("Iteration")
        plt.ylabel("Approximate relative error (%)")
        plt.title("Gauss–Seidel: es(%) vs Iteration")
        plt.grid(True)
        plt.show()

    return x

# ----- example -----
A = [
    [4.0, -1.0,  0],
    [-1.0, 4, -1.0],
    [ 0, -1.0, 4]
]
b = [12, -1, 5]

print("=== Gauss–Seidel (with checks) ===")
xgs = gauss_seidel_method(A, b, tol=1e-3, max_iter=100)
print("Final x (Gauss–Seidel):", xgs)
