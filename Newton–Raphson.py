import math
import matplotlib.pyplot as plt
import sympy as sp

def newton_raphson(func, dfunc, x0, tolerance=1e-6, max_iter=100):
   
    print(f"{'Iter':<5}{'xi':>16}{'xi+1':>16}{'f(xi+1)':>16}{'es(%)':>12}")

    es_series = []
    divergence_counter = 0
    xi = x0

    for i in range(1, max_iter + 1):
        f_xi = func(xi)
        df_xi = dfunc(xi)

        if df_xi == 0:
            print("Derivative is zero. Method fails.")
            return None

        # Newton-Raphson update
        x_next = xi - f_xi / df_xi
        f_next = func(x_next)

        # compute error
        es = abs((x_next - xi) / x_next) * 100 if x_next != 0 else float("inf")

        print(f"{i:<5}{xi:16.10f}{x_next:16.10f}{f_next:16.10f}{es:12.6f}")

        # store error
        es_series.append(es)

        # stopping condition
        if es <= tolerance:
            print("\nConverged successfully.")
            break

        # detect divergence
        if len(es_series) > 1 and es >= es_series[-2]:
            divergence_counter += 1
        else:
            divergence_counter = 0

        if divergence_counter >= 5:
            print("\nMethod is diverging. Exiting early.")
            break

        xi = x_next  # update for next iteration

    print("\nFinal Root Approximation:", f"{x_next:.10f}")
    print("Total Iterations Performed:", i)

    # plot error
    if es_series:
        plt.figure()
        plt.plot(range(1, 1 + len(es_series)), es_series, marker="o")
        plt.xlabel("Iteration")
        plt.ylabel("Approximate relative error (%)")
        plt.title("Newton-Raphson: es(%) vs Iteration")
        plt.grid(True)
        plt.show()

    return x_next

def differentiate(f2):
    x = sp.symbols('x')
    f_sym = f2(x)
    df_sym = sp.diff(f_sym, x)           # Symbolic derivative
    df = sp.lambdify(x, df_sym, 'math')  # Convert to numeric function
    return df
# Example function
def f2(x):
    return (x - 4) * (x - 4) * (x + 2)

# Derivative of f2(x)
df2 = differentiate(f2)
 
def df2(x):
    return 2 * (x - 4) * (x + 2) + (x - 4) * (x - 4)
   

# Run Newton–Raphson
newton_raphson(f2, df2, x0=-2.5, tolerance=0.1)
