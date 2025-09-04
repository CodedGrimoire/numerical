def bisection_method(func, a, b, tol, true_value=None, max_iter=100):
    """
    Bisection Method for root finding.
    
    Parameters:
    func       : function whose root we want to find
    a, b       : interval [a, b]
    tol        : maximum allowed error
    true_value : optional, actual root (if known) to compute true error
    max_iter   : maximum number of iterations
    
    Prints details of each iteration.
    """

    if func(a) * func(b) > 0:
        print("Bisection method fails: f(a) and f(b) must have opposite signs.")
        return None

    print(f"{'Iter':<5}{'Lower':<15}{'Upper':<15}{'Root':<15}{'Approx Error':<20}{'True Error':<20}")
    print("-" * 85)

    root = a
    for i in range(1, max_iter + 1):
        prev_root = root
        root = (a + b) / 2
        f_root = func(root)

        # Approximation error
        approx_error = abs(root - prev_root) if i > 1 else None

        # True error (if true root is given)
        true_error = abs(true_value - root) if true_value is not None else None

        # Print iteration data
        print(f"{i:<5}{a:<15.8f}{b:<15.8f}{root:<15.8f}"
              f"{(approx_error if approx_error is not None else '---')!s:<20}"
              f"{(true_error if true_error is not None else '---')!s:<20}")

        # Check stopping condition
        if approx_error is not None and approx_error < tol:
            break

        # Update interval
        if func(a) * f_root < 0:
            b = root
        else:
            a = root

    print("\nFinal Root Approximation:", root)
    return root


# Example usage
if __name__ == "__main__":
    import math

    # Example: Solve f(x) = x^3 - x - 2 in [1, 2]
    def f(x): return x**3 - x - 2

    true_root = 1.5213797  # known root for comparison
    bisection_method(f, a=1, b=2, tol=1e-6, true_value=true_root)