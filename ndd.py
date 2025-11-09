import matplotlib.pyplot as plt

X_ALL = [0.0, 1.0, 2.5, 3.0, 4.5, 5.0, 6.0]
Y_ALL = [2.00000, 5.43750, 7.35160, 7.56250, 8.44530, 9.18750, 12.00000]
X_TARGET = 3.5


def validate_xy_lengths(x_nodes, y_nodes):
    if len(x_nodes) != len(y_nodes):
        raise ValueError("length mismatch between x and y")


def validate_unique_x(x_nodes):
    seen = set()
    for x in x_nodes:
        if x in seen:
            raise ValueError(f"duplicate x detected: {x}")
        seen.add(x)


def print_data_table():
    print("Given Measured Data")
    print("-" * 35)
    print(f"{'i':>3} {'x_i':>8} {'y_i':>10}")
    print("-" * 35)
    for i, (x, y) in enumerate(zip(X_ALL, Y_ALL)):
        print(f"{i:>3} {x:>8.2f} {y:>10.5f}")
    print("-" * 35)
    print()


def choose_nearest_nodes(x_all, y_all, x_target, k):
    validate_xy_lengths(x_all, y_all)
    validate_unique_x(x_all)
    triples = [(abs(x - x_target), i, x) for i, x in enumerate(x_all)]
    triples.sort(key=lambda t: t[0])
    chosen = triples[:k]
    chosen.sort(key=lambda t: t[2])
    idxs = [t[1] for t in chosen]
    x_nodes = [x_all[i] for i in idxs]
    y_nodes = [y_all[i] for i in idxs]
    return x_nodes, y_nodes


def divided_difference_table(x_points, y_points):
    validate_xy_lengths(x_points, y_points)
    validate_unique_x(x_points)
    n = len(x_points)
    table = []
    table.append(y_points[:])
    for level in range(1, n):
        prev_col = table[level - 1]
        col = []
        for i in range(n - level):
            num = prev_col[i + 1] - prev_col[i]
            den = x_points[i + level] - x_points[i]
            col.append(num / den)
        table.append(col)
    return table


def coeffs_from_table(table):
    return [col[0] for col in table]


def newton_evaluate(x_points, coeffs, x):
    n = len(coeffs)
    result = coeffs[0]
    prod = 1.0
    for i in range(1, n):
        prod *= (x - x_points[i - 1])
        result += coeffs[i] * prod
    return result


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


def newton_to_standard(x_nodes, coeffs):
    poly = [coeffs[0]]
    for i in range(1, len(coeffs)):
        term = [1.0]
        for j in range(i):
            term = poly_mul(term, [-x_nodes[j], 1.0])
        term = [c * coeffs[i] for c in term]
        poly = poly_add(poly, term)
    return poly




def format_power_poly(coeffs, var="x"):
    terms = []
    for i, c in enumerate(coeffs):
        if abs(c) < 1e-12:
            continue
        if i == 0:
            terms.append(f"{c:.10f}")
        elif i == 1:
            terms.append(f"{c:+.10f}{var}")
        else:
            terms.append(f"{c:+.10f}{var}^{i}")
    return " ".join(terms)


def print_divided_diff_table(x_nodes, table):
    n = len(x_nodes)
    print("Divided-Difference Table:")
    header = ["x", "f[x]"]
    for k in range(2, n + 1):
        header.append(f"order {k-1}")
    print(" | ".join(f"{h:>12}" for h in header))
    for i in range(n):
        row = [f"{x_nodes[i]:>12.6f}"]
        for col in range(n):
            if col < len(table) and i < len(table[col]) and i <= (n - col - 1):
                row.append(f"{table[col][i]:>12.8f}")
            else:
                row.append(" " * 12)
        print(" | ".join(row))
    print()


def plot_newton_polys(poly_list, x_range, x_data, y_data):
    xmin, xmax = x_range
    xs = [xmin + (xmax - xmin) * t / 200 for t in range(201)]
    plt.figure()
    plt.scatter(x_data, y_data, label="data points", zorder=5)
    for degree, x_nodes, coeffs in poly_list:
        ys = [newton_evaluate(x_nodes, coeffs, x) for x in xs]
        plt.plot(xs, ys, label=f"N{degree}(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Newton Divided-Difference Polynomials")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_convergence(results, x_target):
    degrees = [deg for deg, *_ in results]
    values = [val for _, _, _, val, _ in results]

    plt.figure()
    plt.plot(degrees, values, "o-", label=f"Pₖ({x_target}) values")
    for i in range(1, len(values)):
        plt.plot([degrees[i-1], degrees[i]],
                 [values[i-1], values[i]], "k--", alpha=0.3)

    plt.xlabel("Polynomial Degree (k-1)")
    plt.ylabel(f"Pₖ({x_target})")
    plt.title(f"Convergence of Pₖ({x_target})")
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    validate_xy_lengths(X_ALL, Y_ALL)
    validate_unique_x(X_ALL)
    print_data_table()
    results = []
    poly_list = []
    for k in range(3, len(X_ALL) + 1):
        x_nodes, y_nodes = choose_nearest_nodes(X_ALL, Y_ALL, X_TARGET, k)
        table = divided_difference_table(x_nodes, y_nodes)
        coeffs = coeffs_from_table(table)
        val = newton_evaluate(x_nodes, coeffs, X_TARGET)
        std_poly = newton_to_standard(x_nodes, coeffs)
        degree = k - 1
        results.append((degree, x_nodes, coeffs, val, std_poly))
        poly_list.append((degree, x_nodes, coeffs))
    print("Task 2 — Newton’s Divided-Difference Polynomial")
    print(f"Target x = {X_TARGET}\n")
    prev_val = None
    for (degree, x_nodes, coeffs, val, std_poly) in results:
        print(f"--- N_{degree}(x) ---")
        print(f"nodes used (by distance from {X_TARGET}): {x_nodes}")
        table = divided_difference_table(x_nodes, [Y_ALL[X_ALL.index(x)] for x in x_nodes])
        print_divided_diff_table(x_nodes, table)
        poly_str = format_power_poly(std_poly, "x")
        print(f"N_{degree}(x) = {poly_str}")
        print(f"N_{degree}({X_TARGET}) = {val:.10f}")
        if prev_val is not None:
            delta = abs(val - prev_val)
            print(f"Δ_{degree} = |N_{degree}({X_TARGET}) - N_{degree-1}({X_TARGET})| = {delta:.10f}")
        else:
            print("Δ_2 = (first Newton interpolant)")
        print()
        prev_val = val
  
    plot_convergence(results, X_TARGET)



if __name__ == "__main__":
    main()
