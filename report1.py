import math
import matplotlib.pyplot as plt

# ------------------- Methods ------------------- #

def false_position_method(func, a, b, tolerance, max_iter=100):
    es_series = []
    if func(a) * func(b) > 0:
        print("\nFalse Position failed: f(a) and f(b) must have opposite signs.")
        return None, es_series

    print("\n=== False Position Method ===")
    xr = (a * func(b) - b * func(a)) / (func(b) - func(a))
    prev_root = None

    for i in range(1, max_iter+1):
        f_a, f_b, f_r = func(a), func(b), func(xr)
        es = None if prev_root is None else abs((xr - prev_root) / xr) * 100
        print(f"Iter {i:<3} xl={a:.6f} xu={b:.6f} xr={xr:.6f} f(xr)={f_r:.6f} es={es if es else '---'}")
        if es: es_series.append(es)
        if es and es <= tolerance: break
        if f_a * f_r < 0: b = xr
        else: a = xr
        prev_root, xr = xr, (a * func(b) - b * func(a)) / (func(b) - func(a))

    print("Final Root:", xr)
    return xr, es_series


def bisection_method(func, a, b, tolerance, max_iter=100):
    es_series = []
    if func(a) * func(b) > 0:
        print("\nBisection failed: f(a) and f(b) must have opposite signs.")
        return None, es_series

    print("\n=== Bisection Method ===")
    prev_root = None
    for i in range(1, max_iter+1):
        xr = (a+b)/2
        f_r = func(xr)
        es = None if prev_root is None else abs((xr - prev_root) / xr) * 100
        print(f"Iter {i:<3} a={a:.6f} b={b:.6f} mid={xr:.6f} f(mid)={f_r:.6f} es={es if es else '---'}")
        if es: es_series.append(es)
        if es and es <= tolerance: break
        if func(a)*f_r < 0: b = xr
        else: a = xr
        prev_root = xr
    print("Final Root:", xr)
    return xr, es_series


def newton_raphson(func, dfunc, x0, tolerance=1e-6, max_iter=100):
    es_series = []
    print("\n=== Newton-Raphson Method ===")
    xi = x0
    for i in range(1, max_iter+1):
        f_x, df_x = func(xi), dfunc(xi)
        if df_x == 0:
            print("Derivative zero. Failed.")
            return None, es_series
        x_next = xi - f_x/df_x
        es = abs((x_next - xi)/x_next)*100
        print(f"Iter {i:<3} xi={xi:.6f} xi+1={x_next:.6f} f(xi+1)={func(x_next):.6f} es={es:.6f}")
        es_series.append(es)
        if es <= tolerance: break
        xi = x_next
    print("Final Root:", x_next)
    return x_next, es_series


def secant_method(func, x0, x1, tolerance, max_iter=100):
    es_series = []
    print("\n=== Secant Method ===")
    for i in range(1, max_iter+1):
        f0, f1 = func(x0), func(x1)
        if f1-f0 == 0:
            print("Division by zero. Failed.")
            return None, es_series
        x2 = x1 - f1*(x1-x0)/(f1-f0)
        es = abs((x2-x1)/x2)*100
        print(f"Iter {i:<3} x0={x0:.6f} x1={x1:.6f} x2={x2:.6f} f(x2)={func(x2):.6f} es={es:.6f}")
        es_series.append(es)
        if es <= tolerance: break
        x0, x1 = x1, x2
    print("Final Root:", x2)
    return x2, es_series

# ------------------- Function ------------------- #

def f2(x):
    return x**3 - 10*x + 5*math.exp(-x/2) - 2

def df2(x):
    return 3*x**2 - 10 - (5/2)*math.exp(-x/2)

# ------------------- Run All Methods ------------------- #

fp_root, fp_err = false_position_method(f2, .1, .4, .001)
bi_root, bi_err = bisection_method(f2, .1, .4, .001)
nr_root, nr_err = newton_raphson(f2, df2, 1.5,.001)
sc_root, sc_err = secant_method(f2, 1.5, 2, .001)

# ------------------- Combined Plot ------------------- #

plt.figure(figsize=(9,6))
plt.plot(range(1, len(fp_err)+1), fp_err, marker='o', label="False Position")
plt.plot(range(1, len(bi_err)+1), bi_err, marker='o', label="Bisection")
plt.plot(range(1, len(nr_err)+1), nr_err, marker='o', label="Newton-Raphson")
plt.plot(range(1, len(sc_err)+1), sc_err, marker='o', label="Secant")

plt.xlabel("Iteration", fontsize=12)
plt.ylabel("Approximate Relative Error (%)", fontsize=12)
plt.title("Error Convergence of Root-Finding Methods", fontsize=14, weight="bold")
plt.yscale("log")  # makes comparison clearer

plt.legend()
plt.tight_layout()
plt.show()
