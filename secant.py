import math
import matplotlib.pyplot as plt

def secant_method(func, x0, x1, tolerance, max_iter=100):
    print(f"{'Iter':<5}{'xi-1':>16}{'xi':>16}{'xi+1':>16}{'f(xi+1)':>16}{'es(%)':>12}")

    guesses = [x0, x1]
    es_series = []
    divergence_counter = 0

    for i in range(1, max_iter + 1):
        f_x0 = func(x0)
        f_x1 = func(x1)

        if f_x1 - f_x0 == 0:
            print("Division by zero. Method fails.")
            return None

        # Secant update
        x_next = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        f_next = func(x_next)

        # compute error
        es = abs((x_next - x1) / x_next) * 100 if x_next != 0 else float("inf")

        print(f"{i:<5}{x0:16.10f}{x1:16.10f}{x_next:16.10f}{f_next:16.10f}{es:12.6f}")

        guesses.append(x_next)

        if es != float("inf"):
            es_series.append(es)

        # stopping condition
        if es <= tolerance:
            print("\nConverged successfully.")
            break

        # detect divergence
        if len(guesses) > 2 and es >= abs((guesses[-2] - guesses[-3]) / guesses[-2]) * 100:
            divergence_counter += 1
        else:
            divergence_counter = 0

        if divergence_counter >= 5:
            print("\nMethod is diverging. Exiting early.")
            break

        # shift for next iteration
        x0, x1 = x1, x_next

    print("\nFinal Root Approximation:", f"{x_next:.10f}")
    print("Total Iterations Performed:", i)

    # Plot iteration vs error
    if es_series:
        plt.figure()
        plt.plot(range(1, len(es_series)+1), es_series, marker='o')
        plt.xlabel("Iteration")
        plt.ylabel("Approximate relative error (%)")
        plt.title("Secant Method: Iteration vs Error")
        plt.grid(True)
        plt.show()

    return x_next


# Example function
def f2(x):
    return (x - 4) * (x - 4) * (x + 2)

# Run Secant Method
secant_method(f2, x0=-2.5, x1=-1.5, tolerance=0.1)
