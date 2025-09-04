import sympy as sp

def taylor_series(func, a, x_val, tol, max_terms=50):
    """
    Approximates a function using Taylor series expansion around a point 'a'
    
    Parameters:
    func     : sympy expression of the function
    a        : point of expansion (center of Taylor series)
    x_val    : value at which function is approximated
    tol      : tolerance (stop when next term < tol)
    max_terms: maximum number of terms
    
    Returns:
    approximation of f(x_val)
    """
    x = sp.Symbol('x')
    f = func

    # true value
    true_value = f.subs(x, x_val).evalf()

    # initialization
    series_sum = 0
    factorial = 1
    print(f"{'Term':<5}{'Derivative':<20}{'Term Value':<20}{'Partial Sum':<20}{'Error':<20}")
    print("-" * 90)

    for n in range(max_terms):
        # nth derivative
        derivative = f.diff(x, n)
        derivative_at_a = derivative.subs(x, a).evalf()

        # nth term
        term_value = (derivative_at_a * ((x_val - a) ** n)) / sp.factorial(n)
        series_sum += term_value

        # error (true error)
        error = abs(true_value - series_sum)

        print(f"{n:<5}{derivative_at_a:<20}{term_value:<20}{series_sum:<20}{error:<20}")

        # stopping condition
        if abs(term_value) < tol:
            break

    print(f"\nFinal Approximation of f({x_val}) = {series_sum}")
    print(f"True Value f({x_val}) = {true_value}")
    print(f"Final Error = {error}")
    return series_sum


# Example usage
if __name__ == "__main__":
    x = sp.Symbol('x')
    f = sp.exp(x)   # example function: e^x
    taylor_series(f, a=0, x_val=1, tol=1e-6)