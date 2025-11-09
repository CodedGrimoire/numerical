import matplotlib.pyplot as plt

# ----------------------------
# given measured data
# ----------------------------
X_ALL = [0.0, 1.0, 2.5, 3.0, 4.5, 5.0, 6.0]
Y_ALL = [2.00000, 5.43750, 7.35160, 7.56250, 8.44530, 9.18750, 12.00000]
X_TARGET = 3.5


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
# node selection: nearest to target
# ----------------------------
def choose_nearest_nodes(x_all, y_all, x_target, k):
    # (distance, original_index, x_value)
    triples = [(abs(x - x_target), i, x) for i, x in enumerate(x_all)]
    triples.sort(key=lambda t: t[0])      # nearest first
    chosen = triples[:k]
    # for Newton, we want them in x-order (makes the table nice)
    chosen.sort(key=lambda t: t[2])
    idxs = [t[1] for t in chosen]
    x_nodes = [x_all[i] for i in idxs]
    y_nodes = [y_all[i] for i in idxs]
    return x_nodes, y_nodes


# ----------------------------
# divided difference table
# ----------------------------
def divided_difference_table(x_points, y_points):
    """
    returns the full table as a list of columns
    table[0] = f[x0], f[x1], ...
    table[1] = f[x0,x1], f[x1,x2], ...
    ...
    """
    n = len(x_points)
    table = []
    # column 0: f[x_i]
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
    """Newton coefficients are the top elements of each column"""
    return [col[0] for col in table]


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


# ----------------------------
# evaluate Newton polynomial
# ----------------------------
def newton_evaluate(x_points, coeffs, x):
    """
    P(x) = c0 + c1 (x - x0) + c2 (x - x0)(x - x1) + ...
    """
    n = len(coeffs)
    result = coeffs[0]
    prod = 1.0
    for i in range(1, n):
        prod *= (x - x_points[i - 1])
        result += coeffs[i] * prod
    return result


# ----------------------------
# plotting multiple Newton polynomials
# ----------------------------
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


def main():
    print_data_table()

    results = []   # (degree, x_nodes, coeffs, value_at_target)
    poly_list = [] # for plotting

    # like task 1: start with 3 nodes (quadratic, degree=2) up to all 7 nodes (degree=6)
    for k in range(3, len(X_ALL) + 1):
        x_nodes, y_nodes = choose_nearest_nodes(X_ALL, Y_ALL, X_TARGET, k)

        # 1) build full table
        table = divided_difference_table(x_nodes, y_nodes)

        # 2) get coefficients
        coeffs = coeffs_from_table(table)

        # 3) evaluate at target
        val = newton_evaluate(x_nodes, coeffs, X_TARGET)

        degree = k - 1
        results.append((degree, x_nodes, coeffs, val, table))
        poly_list.append((degree, x_nodes, coeffs))

    print("Task 2 — Newton’s Divided-Difference Polynomial")
    print(f"Target x = {X_TARGET}\n")

    prev_val = None
    for (degree, x_nodes, coeffs, val, table) in results:
        print(f"--- N_{degree}(x) ---")
        print(f"nodes used (by distance from {X_TARGET}): {x_nodes}")
        print_divided_diff_table(x_nodes, table)
        print("Newton form:")
        # print N_k(x) = c0 + c1(x - x0) + c2(x - x0)(x - x1) + ...
        terms = [f"{coeffs[0]:.8f}"]
        for i in range(1, len(coeffs)):
            factors = "".join([f"(x - {x_nodes[j]:.4f})" for j in range(i)])
            terms.append(f"{coeffs[i]:+.8f}{factors}")
        print("N_{:d}(x) = ".format(degree) + " ".join(terms))
        print(f"N_{degree}({X_TARGET}) = {val:.10f}")
        if prev_val is not None:
            delta = abs(val - prev_val)
            print(f"Δ_{degree} = |N_{degree}({X_TARGET}) - N_{degree-1}({X_TARGET})| = {delta:.10f}")
        else:
            print("Δ_2 = (first Newton interpolant)")
        print()

        prev_val = val

    # plot all
    plot_newton_polys(poly_list, (0.0, 6.0), X_ALL, Y_ALL)


if __name__ == "__main__":
    main()
