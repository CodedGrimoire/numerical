import matplotlib.pyplot as plt

# ----------------------------
# given measured data
# ----------------------------
X_ALL = [0.0, 1.0, 2.5, 3.0, 4.5, 5.0, 6.0]
Y_ALL = [2.00000, 5.43750, 7.35160, 7.56250, 8.44530, 9.18750, 12.00000]
X_TARGET = 3.5  # the x where we evaluate Pk(x)


def print_data_table():
    print("Given Measured Data")
    print("-" * 35)
    print(f"{'i':>3} {'x_i':>8} {'y_i':>10}")
    print("-" * 35)
    for i, (x, y) in enumerate(zip(X_ALL, Y_ALL)):
        print(f"{i:>3} {x:>8.2f} {y:>10.5f}")
    print("-" * 35)
    print()


# ----------------------------
# core lagrange utilities
# ----------------------------
def lagrange_value(x_nodes, y_nodes, x):
    n = len(x_nodes)
    total = 0.0
    for i in range(n):
        xi = x_nodes[i]
        Li = 1.0
        for j in range(n):
            if j != i:
                Li *= (x - x_nodes[j]) / (xi - x_nodes[j])
        total += y_nodes[i] * Li
    return total


def lagrange_poly_coeffs(x_nodes, y_nodes):
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
            Li = poly_mul(Li, [-xj, 1.0])  # (x - xj)
            denom *= (xi - xj)
        Li = [coef * (y_nodes[i] / denom) for coef in Li]
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


# ----------------------------
# helper: choose k nearest nodes to target
# ----------------------------
def choose_nearest_nodes(x_all, y_all, x_target, k):
    triples = [(abs(x - x_target), i, x) for i, x in enumerate(x_all)]
    triples.sort(key=lambda t: t[0])  # nearest first
    chosen = triples[:k]
    chosen.sort(key=lambda t: t[2])   # sort by x-value
    idxs = [t[1] for t in chosen]
    x_nodes = [x_all[i] for i in idxs]
    y_nodes = [y_all[i] for i in idxs]
    return x_nodes, y_nodes, idxs


# ----------------------------
# plotting
# ----------------------------
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


# ----------------------------
# main according to lab task
# ----------------------------
def main():
    print_data_table()

    results = []  # (degree, x_nodes, coeffs, value_at_3_5)

    # degrees from 2 (3 nodes) up to 6 (7 nodes)
    for k in range(3, len(X_ALL) + 1):
        x_nodes, y_nodes, idxs = choose_nearest_nodes(X_ALL, Y_ALL, X_TARGET, k)
        degree = k - 1

        coeffs = lagrange_poly_coeffs(x_nodes, y_nodes)
        val = poly_eval(coeffs, X_TARGET)

        results.append((degree, x_nodes, coeffs, val))

    print("Task 1 — Lagrange Interpolating Polynomial")
    print(f"Target x = {X_TARGET}\n")

    prev_val = None
    poly_for_plot = []

    for (degree, x_nodes, coeffs, val) in results:
        print(f"--- P_{degree}(x) ---")
        print(f"nodes used (closest to {X_TARGET}): {x_nodes}")
        print("polynomial coefficients (c0 + c1 x + c2 x^2 + ...):")
        for i, c in enumerate(coeffs):
            print(f"  c{i} = {c:.10f}")
        print(f"P_{degree}({X_TARGET}) = {val:.10f}")
        if prev_val is not None:
            delta = val - prev_val
            print(f"Δ_{degree} = P_{degree}({X_TARGET}) - P_{degree-1}({X_TARGET}) = {delta:.10f}")
        else:
            print("Δ_2 = (first interpolant)")
        print()

        prev_val = val
        poly_for_plot.append((degree, coeffs))

    plot_all_polynomials(poly_for_plot, (0.0, 6.0), X_ALL, Y_ALL)


if __name__ == "__main__":
    main()
