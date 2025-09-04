import math
import matplotlib.pyplot as plt

def bisection_method(func, a, b, tol, max_iter=100):
    tolerance=tol/100
    if func(a) * func(b) > 0:
        print("Bisection method fails")
        return None

    print(f"{'Iter':<5}{'xl':>16}{'xu':>16}{'xr':>16}{'f(xr)':>16}{'es(%)':>12}")

    prev_root = None
    es_series = []

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

       
        if es is not None and es <= tolerance:
            break

       
        if func(a) * f_root < 0:
            b = root
        else:
            a = root

        
        prev_root = root
        root = (a + b) / 2.0

        # store error for plotting (skip None on first iter)
        if es is not None:
            es_series.append(es)

    print("\nFinal Root Approximation:", f"{root:.10f}")
    print("Total Iterations Performed:", i)

    # Plot approximate error (%) vs iteration
    if es_series:
        plt.figure()
        plt.plot(range(2, 2 + len(es_series)), es_series, marker='o')
        plt.xlabel("Iteration")
        plt.ylabel("Approximate relative error (%)")
        plt.title("Bisection: es(%) vs Iteration")
        plt.grid(True)
        plt.show()

    return root


def f2(x): 
    return 225 + 82*x - 90*x**2 + 44*x**3 - 8*x**4 + 0.7*x**5


bisection_method(f2, a=-1.2, b=-1.0, tol=5)
