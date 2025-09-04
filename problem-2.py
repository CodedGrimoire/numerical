import math
import matplotlib.pyplot as plt

def bisection_method(func, a, b, tolerance, max_iter=100):
    if func(a) * func(b) > 0:
        print("Bisection method fails")
        return None

    print(f"{'Iter':<5}{'xl':>16}{'xu':>16}{'xr':>16}{'f(xr)':>16}{'es(%)':>12}")

    prev_root = None
    es_series = []
    iter_series = []

    root = (a + b) / 2.0

    for i in range(1, max_iter + 1):
        f_root = func(root)

        if prev_root is None:
            es = None
        else:
            es = abs((root - prev_root) / root) * 100 if root != 0 else float('inf')

        print(
            f"{i:<5}"
            f"{a:16.10f}{b:16.10f}{root:16.10f}{f_root:16.10f}"
            f"{(f'{es:.6f}' if es is not None else '---'):>12}"
        )

        if es is not None:
            es_series.append(es)
            iter_series.append(i)

        if es is not None and es <= tolerance:
            break

        if func(a) * f_root < 0:
            b = root
        else:
            a = root

        prev_root = root
        root = (a + b) / 2.0

    print("\nFinal Root Approximation:", f"{root:.6f}")
    print("Total Iterations Performed:", i)

    if es_series:
        plt.figure()
        plt.plot(iter_series, es_series, marker='o')
        plt.xlabel("Iteration")
        plt.ylabel("Approximate relative error (%)")
        plt.title("Bisection: es(%) vs iteration")
        plt.grid(True)
        plt.show()

    return root

Q = 20.0
g = 9.81

def B(y):
    return 3.0 + y

def Ac(y):
    return 3.0*y + 0.5*y*y

def f_channel(y):
    A = Ac(y)
    return 1.0 - (Q*Q) / (g * (A**3) * B(y))

xl, xu = 0.5, 2.5
bisection_method(f_channel, a=xl, b=xu, tolerance=1, max_iter=10)
