import math
import matplotlib.pyplot as plt


EPS_PIVOT = 1e-14

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


def validate_inputs(A, b, x0, tol, max_iter):
    if not isinstance(A, list) or not all(isinstance(row, list) for row in A):
        print("Error: A must be a list of lists."); return False
    n = len(A)
    if n == 0:
        print("Error: A must be non-empty."); return False
    if any(len(row) != n for row in A):
        print("Error: A must be square."); return False
    if not isinstance(b, list) or len(b) != n:
        print("Error: b must be a list of length equal to len(A)."); return False
    if not isinstance(x0, list) or len(x0) != n:
        print("Error: initial guess must be a list of length equal to len(A)."); return False
    try:
        for i in range(n):
            for j in range(n):
                float(A[i][j])
            float(b[i]); float(x0[i])
    except Exception:
        print("Error: A, b, and x0 must contain numeric values."); return False
    try:
        tol = float(tol)
    except Exception:
        print("Error: tol must be a positive number."); return False
    if tol <= 0:
        print("Error: tol must be > 0."); return False
    if not isinstance(max_iter, int) or max_iter < 1:
        print("Error: max_iter must be an integer ≥ 1."); return False
    return True

def det_via_elimination(Ain):
    A = [row[:] for row in Ain]
    n = len(A)
    det = 1.0
    swap_count = 0
    for i in range(n):
        piv = i
        maxabs = abs(A[i][i])
        for r in range(i+1, n):
            if abs(A[r][i]) > maxabs:
                maxabs = abs(A[r][i]); piv = r
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

def matvec(A, x):
    return [sum(A[i][j]*x[j] for j in range(len(x))) for i in range(len(A))]

def residual_l2(A, x, b):
    r = [matvec(A, x)[i] - b[i] for i in range(len(b))]
    return math.sqrt(sum(ri*ri for ri in r))


def jacobi_method(A_in, b_in, x0, tol, max_iter=100, diverge_patience=5):
    A = [row[:] for row in A_in]
    b = b_in[:]
    n = len(b)
    if any(len(row) != n for row in A):
        print("Error: A must be square and match b."); return None, []
    d = det_via_elimination(A)
    if abs(d) < EPS_PIVOT:
        print("Jacobi aborted: det(A) ≈ 0 (singular/ill-conditioned)."); return None, []
    for i in range(n):
        if abs(A[i][i]) < EPS_PIVOT:
            print(f"Jacobi aborted: A[{i},{i}] is zero/near-zero; cannot proceed without row swaps."); return None, []
    if not is_strictly_diag_dominant(A):
        print("Warning: A is not strictly diagonally dominant; Jacobi may diverge or be slow.")
    x_old = x0[:]
    res_series = []
    worse_streak = 0
    header = f"{'Iter':<5}" + "".join([f"{('x'+str(i+1)):>12}" for i in range(n)]) + "".join([f"{('e'+str(i+1)):>12}" for i in range(n)]) + f"{'res(L2)':>14}"
    print(header)
    converged = False
    for it in range(1, max_iter+1):
        x_new = [0.0]*n
        for i in range(n):
            s = 0.0
            for j in range(n):
                if j != i:
                    s += A[i][j] * x_old[j]
            x_new[i] = (b[i] - s) / A[i][i]
        abs_errs = [abs(x_new[i]-x_old[i]) for i in range(n)]
        res = residual_l2(A, x_new, b)
        res_series.append(res)
        row = f"{it:<5}" + "".join([f"{x_new[i]:12.10f}" for i in range(n)]) + "".join([f"{abs_errs[i]:12.10f}" for i in range(n)]) + f"{res:14.10f}"
        print(row)
        if it >= 2 and res >= res_series[-2] - 1e-12:
            worse_streak += 1
        else:
            worse_streak = 0
        if worse_streak >= diverge_patience:
            print("\nJacobi stopping early: residual not improving."); break
        x_old = x_new
        if res <= tol:
            print("\nConverged by residual (Jacobi)."); converged = True; break
    if not converged:
        print("\nDid not converge within the maximum iterations (Jacobi).")
    return x_old, res_series



def gauss_seidel_method(A_in, b_in, x0, tol, max_iter=100, diverge_patience=5):
    A = [row[:] for row in A_in]
    b = b_in[:]
    n = len(b)
    if any(len(row) != n for row in A):
        print("Error: A must be square and match b."); return None, []
    d = det_via_elimination(A)
    if abs(d) < EPS_PIVOT:
        print("Gauss–Seidel aborted: det(A) ≈ 0 (singular or ill-conditioned)."); return None, []
    if not try_fix_zero_diagonal(A, b):
        print("Gauss–Seidel aborted: zero/near-zero diagonal could not be fixed by row swap."); return None, []
    if not is_strictly_diag_dominant(A):
        print("Warning: A is not strictly diagonally dominant; Gauss–Seidel may diverge or be slow.")
    x = x0[:]
    res_series = []
    worse_streak = 0
    header = f"{'Iter':<5}" + "".join([f"{('x'+str(i+1)):>12}" for i in range(n)]) + "".join([f"{('e'+str(i+1)):>12}" for i in range(n)]) + f"{'res(L2)':>14}"
    print(header)
    converged = False
    for it in range(1, max_iter+1):
        x_prev = x[:]
        for i in range(n):
            s1 = sum(A[i][j]*x[j] for j in range(0, i))
            s2 = sum(A[i][j]*x_prev[j] for j in range(i+1, n))
            x[i] = (b[i] - s1 - s2) / A[i][i]
        abs_errs = [abs(x[i]-x_prev[i]) for i in range(n)]
        res = residual_l2(A, x, b)
        res_series.append(res)
        row = f"{it:<5}" + "".join([f"{x[i]:12.10f}" for i in range(n)]) + "".join([f"{abs_errs[i]:12.10f}" for i in range(n)]) + f"{res:14.10f}"
        print(row)
        if it >= 2 and res >= res_series[-2] - 1e-12:
            worse_streak += 1
        else:
            worse_streak = 0
        if worse_streak >= diverge_patience:
            print("\nGauss–Seidel stopping early: residual not improving."); break
        if res <= tol:
            print("\nConverged by residual (Gauss–Seidel)."); converged = True; break
    if not converged:
        print("\nDid not converge within the maximum iterations (Gauss–Seidel).")
    return x, res_series

def read_matrix(n, prompt):
    print(prompt)
    A = []
    for i in range(n):
        while True:
            line = input().strip()
            if line == "": continue
            parts = line.split()
            if len(parts) != n:
                print(f"Error: need exactly {n} numbers. Re-enter row {i+1}:")
                continue
            try:
                row = [float(x) for x in parts]
                A.append(row)
                break
            except:
                print("Error: non-numeric value. Re-enter row:")
    return A

def read_vector(n, prompt):
    print(prompt)
    v = []
    for i in range(n):
        while True:
            line = input().strip()
            if line == "": continue
            try:
                v.append(float(line))
                break
            except:
                print("Error: enter a numeric value:")
    return v


def main():
    try:
        n = int(input("Enter number of equations: ").strip())
    except:
        print("Error: invalid integer for number of equations."); return
    A = read_matrix(n, "Enter coefficient matrix A (row-wise):")
    b = read_vector(n, "Enter constants vector b:")
    x0 = read_vector(n, "Enter initial guess vector:")
    try:
        max_iter = int(input("Enter maximum iterations: ").strip())
    except:
        print("Error: invalid integer for maximum iterations."); return
    try:
        tol = float(input("Enter tolerance: ").strip())
    except:
        print("Error: invalid number for tolerance."); return
    if not validate_inputs(A, b, x0, tol, max_iter):
        return

    print("\n=== Gauss–Seidel (residual-based) ===")
    xgs, res_gs = gauss_seidel_method(A, b, x0, tol=tol, max_iter=max_iter)
    print("Final x (Gauss–Seidel):", xgs)

    print("\n=== Jacobi (residual-based, no swap) ===")
    xj, res_j = jacobi_method(A, b, x0, tol=tol, max_iter=max_iter)
    print("Final x (Jacobi):", xj)

    if res_j or res_gs:
        plt.figure()
        if res_j:
            plt.plot(range(1, len(res_j)+1), res_j, marker='o', label='Jacobi residual L2')
        if res_gs:
            plt.plot(range(1, len(res_gs)+1), res_gs, marker='o', label='Gauss–Seidel residual L2')
        plt.xlabel("Iteration")
        plt.ylabel("Residual L2 norm (||Ax - b||2)")
        plt.title("Residual vs Iteration (Jacobi vs Gauss–Seidel)")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()
