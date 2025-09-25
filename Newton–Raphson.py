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

        
        x_next = xi - f_xi / df_xi
        f_next = func(x_next)

        # compute error
        es = abs((x_next - xi) / x_next) * 100 if x_next != 0 else float("inf")

        print(f"{i:<5}{xi:16.10f}{x_next:16.10f}{f_next:16.10f}\t{es:12.10f}")

        # store error
        es_series.append(es)

        # stopping condition
        if es <= tolerance:
            print("\nConverged successfully.")
            break

        
        if len(es_series) > 1 and es >= es_series[-2]:
            divergence_counter += 1
        else:
            divergence_counter = 0

        if divergence_counter >= 5:
            print("\nMethod is diverging. Exiting early.")
            break

        xi = x_next  

    print("\nFinal Root Approximation:", f"{x_next:.10f}")
    print("Total Iterations Performed:", i)

    
    if es_series:
        plt.figure()
        plt.plot(range(1, 1 + len(es_series)), es_series, marker="o")
        plt.xlabel("Iteration")
        plt.ylabel("Approximate relative error (%)")
        plt.title("Newton-Raphson: es(%) vs Iteration")
        plt.grid(True)
        plt.show()

    return x_next




def f2(x):
    return  ((9.81*x)/15)*(1-math.exp((-15/x)*10))-36

# Derivative of f2(x)
#df2 = differentiate(f2)
 
def df2(m):
    return (9.81/15) * (1 - math.exp(-150/m) - (150/m) * math.exp(-150/m))

   


newton_raphson(f2, df2, x0=40, tolerance=0.001)
