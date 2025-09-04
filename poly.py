def poly_taylor(coeffs, a, x_val, tol=1e-6):
    """
    Approximate a polynomial using Taylor series expansion around 'a'.

    Parameters:
    coeffs : list of coefficients [c0, c1, c2, ...] for polynomial c0 + c1*x + c2*x^2 + ...
    a      : point of expansion
    x_val  : value to approximate at
    tol    : error tolerance

    Returns:
    approximation of f(x_val)
    """
    # Define the polynomial function
    def poly(x):
        return sum(c * (x ** i) for i, c in enumerate(coeffs))

    true_value = poly(x_val)
    n = len(coeffs) - 1  # degree of polynomial

    series_sum = 0
    print(f"{'Term':<5}{'Coefficient':<15}{'Term Value':<20}{'Partial Sum':<20}{'Error':<20}")
    print("-" * 85)

    for i, c in enumerate(coeffs):
        # ith derivative at 'a' for polynomials is straightforward
        # For polynomials, Taylor expansion is exact after degree n
        term_value = c * ((x_val - a) ** i)
        series_sum += term_value

        error = abs(true_value - series_sum)

        print(f"{i:<5}{c:<15}{term_value:<20}{series_sum:<20}{error:<20}")

        if error < tol:
            break

    print(f"\nFinal Approximation of f({x_val}) = {series_sum}")
    print(f"True Value f({x_val}) = {true_value}")
    print(f"Final Error = {error}")
    return series_sum


# Example usage: f(x) = 2 + 3x + x^2  aq    qwertyt x=2, expansion at a=0
coeffs = [2, 3, 1]   # 2 + 3x + 1*x^2 x
poly_taylor(coeffs, a=0, x_val=2, tol=1e-6)