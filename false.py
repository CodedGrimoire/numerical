import math
import matplotlib.pyplot as plt

def false_position_method(func, lower, upper, tolerance, max_iter=100):
    a=lower
    b=upper
    if func(a) * func(b) > 0:
        print("False Position method fails")
        return None

    print(f"{'Iter':<5}{'xl':>16}{'xu':>16}{'xr':>16}{'f(xr)':>16}{'es(%)':>12}")

    prev_root = None
    es_series = []
    divergence_counter = 0

    
    root = (a * func(b) - b * func(a)) / (func(b) - func(a))

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

        # store error before checking stop condition
        if es is not None:
            es_series.append(es)

        # stopping condition
        if es is not None and es <= tolerance:
            print("\nConverged successfully.")
            break

        # detect divergence
        if es is not None:
            if len(es_series) > 1 and es >= es_series[-2]:#the second-to-last element.
                divergence_counter += 1
            else:
                divergence_counter = 0

            if divergence_counter >= 5:
                print("\nMethod is diverging. Exiting early.")
                break

        # update interval
        if func(a) * f_root < 0:
            b = root
        else:
            a = root

        prev_root = root
        # false position update
        root = (a * func(b) - b * func(a)) / (func(b) - func(a))

    print("\nFinal Root Approximation:", f"{root:.10f}")
    print("Total Iterations Performed:", i)

    # Plot approximate error (%) vs iteration
    if es_series:
        plt.figure()
        plt.plot(range(2, 2 + len(es_series)), es_series, marker='o')
        plt.xlabel("Iteration")
        plt.ylabel("Approximate relative error (%)")
        plt.title("False Position: es(%) vs Iteration")
        plt.grid(True)
        plt.show()

    return root


def f2(x): 
    return (x-4)*(x-4)*(x+2)


false_position_method(f2, lower=-2.5, upper=-1.0, tolerance=.1)
