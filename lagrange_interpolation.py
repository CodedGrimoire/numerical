import matplotlib.pyplot as plt

X_ALL = [0.0, 1.0, 2.5, 3.0, 4.5, 5.0, 6.0]
Y_ALL = [2.00000, 5.43750, 7.35160, 7.56250, 8.44530, 9.18750, 12.00000]
X_TARGET = 3.5


def validate_unique_x(x_nodes, context="x_nodes"):
    seen = set()
    for x in x_nodes:
        if x in seen:
            raise ValueError(f"Duplicate x value detected in {context}: {x}")
        seen.add(x)


def validate_xy_lengths(x_nodes, y_nodes, context="(x_nodes, y_nodes)"):
    if len(x_nodes) != len(y_nodes):
        raise ValueError(
            f"Mismatched lengths in {context}: "
            f"{len(x_nodes)} x-values, {len(y_nodes)} y-values."
        )


def print_data_table(x_all, y_all):
    print("Given Measured Data")
    print("-" * 35)
    print(f"{'i':>3} {'x_i':>8} {'y_i':>10}")
    print("-" * 35)
    for i, (x, y) in enumerate(zip(x_all, y_all)):
        print(f"{i:>3} {x:>8.2f} {y:>10.5f}")
    print("-" * 35)
    print()


def lagrange_poly_coeffs(x_nodes, y_nodes):
    validate_xy_lengths(x_nodes, y_nodes, context="lagrange_poly_coeffs")
    validate_unique_x(x_nodes, context="lagrange_poly_coeffs")
    n = len(x_nodes)
    coeffs = [0.0] * n
    for i in range(n):
        Li = [1.0]
        denom = 1.0
        xi = x_nodes[i]
        for j in range(n):
            if j == i:
                continue
            xj = x_nodes[j]
            Li = poly_mul(Li, [-xj, 1.0])
            denom *= (xi - xj)
        scale = y_nodes[i] / denom
        Li = [coef * scale for coef in Li]
        coeffs = poly_add(coeffs, Li)
    return coeffs


def poly_add(p, q):
    m = max(len(p), len(q))
    res = [0.0] * m
    for i in range(m):
        a = p[i] if i < len(p) else 0.0
        b = q[i] if i < len(q) else 0.0
        res[i] = a + b
    return res


def poly_mul(p, q):
    res = [0.0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            res[i + j] += a * b
    return res


def poly_eval(coeffs, x):
    total = 0.0
    power = 1.0
    for c in coeffs:
        total += c * power
        power *= x
    return total


def choose_nearest_nodes(x_all, y_all, x_target, k):
    validate_xy_lengths(x_all, y_all, context="choose_nearest_nodes")
    validate_unique_x(x_all, context="choose_nearest_nodes")
    triples = [(abs(x - x_target), i, x) for i, x in enumerate(x_all)]
    triples.sort(key=lambda t: t[0])
    chosen = triples[:k]
    chosen.sort(key=lambda t: t[2])
    idxs = [t[1] for t in chosen]
    x_nodes = [x_all[i] for i in idxs]
    y_nodes = [y_all[i] for i in idxs]
    return x_nodes, y_nodes, idxs


def plot_all_polynomials(polys, x_range, x_data, y_data):
    xmin, xmax = x_range
    xs = [xmin + (xmax - xmin) * t / 200 for t in range(201)]
    plt.figure()
    plt.scatter(x_data, y_data, label="data points", zorder=5)
    for degree, coeffs in polys:
        ys = [poly_eval(coeffs, x) for x in xs]
        plt.plot(xs, ys, label=f"P{degree}(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Lagrange Interpolants Pk(x)")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_convergence(results, x_target):
    degrees = [deg for deg, *_ in results]
    values = [val for *_, val in results]
    plt.figure()
    plt.plot(degrees, values, "o-", label=f"Pₖ({x_target})")
    plt.xlabel("Polynomial degree (k)")
    plt.ylabel(f"Pₖ({x_target})")
    plt.title(f"Convergence of Pₖ({x_target})")
    plt.grid(True)
    plt.legend()
    plt.show()


def format_polynomial(coeffs, var="x", tol=1e-10):
    terms = []
    for power, c in enumerate(coeffs):
        if abs(c) < tol:
            continue
        if power == 0:
            term = f"{c:.10f}"
        elif power == 1:
            term = f"{c:.10f}{var}"
        else:
            term = f"{c:.10f}{var}^{power}"
        terms.append(term)
    if not terms:
        return "0"
    expression = terms[0]
    for term in terms[1:]:
        if term.startswith("-"):
            expression += " - " + term[1:]
        else:
            expression += " + " + term
    return expression


def run_interpolation(x_all, y_all, x_target):
    validate_xy_lengths(x_all, y_all, context="(x_all, y_all)")
    validate_unique_x(x_all, context="x_all")
    print_data_table(x_all, y_all)
    results = []
    poly_for_plot = []
    for k in range(3, len(x_all) + 1):
        x_nodes, y_nodes, idxs = choose_nearest_nodes(x_all, y_all, x_target, k)
        degree = k - 1
        coeffs = lagrange_poly_coeffs(x_nodes, y_nodes)
        val = poly_eval(coeffs, x_target)
        results.append((degree, x_nodes, coeffs, val))
        poly_for_plot.append((degree, coeffs))
    print("Task 1 — Lagrange Interpolating Polynomial")
    print(f"Target x = {x_target}\n")
    prev_val = None
    for (degree, x_nodes, coeffs, val) in results:
        print(f"--- P_{degree}(x) ---")
        print(f"nodes used (closest to {x_target}): {x_nodes}")
        poly_str = format_polynomial(coeffs, var="x")
        print(f"P_{degree}(x) = {poly_str}")
        print(f"P_{degree}({x_target}) = {val:.10f}")
        if prev_val is not None:
            delta = val - prev_val
            print(
                f"Δ_{degree} = P_{degree}({x_target}) - "
                f"P_{degree-1}({x_target}) = {delta:.10f}"
            )
        else:
            print("Δ_2 = (first interpolant)")
        print()
        prev_val = val
    plot_convergence(results, x_target)
   


def main():
    run_interpolation(X_ALL, Y_ALL, X_TARGET)


if __name__ == "__main__":
    main()
