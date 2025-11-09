import matplotlib.pyplot as plt

# -------------------------------------------------
# sensor data along 100 km highway
# -------------------------------------------------
X_ALL = [0, 10, 20, 35, 50, 65, 80, 90, 100]
T_ALL = [25.0, 26.7, 29.4, 33.2, 35.5, 36.1, 37.8, 38.9, 40.0]
X_TARGET = 45  # km


def print_data_table():
    print("Highway Temperature Data (at 12:00 PM)")
    print("-" * 45)
    print(f"{'i':>3} {'Distance(km)':>14} {'Temperature(°C)':>18}")
    print("-" * 45)
    for i, (x, t) in enumerate(zip(X_ALL, T_ALL)):
        print(f"{i:>3} {x:>14.1f} {t:>18.1f}")
    print("-" * 45)
    print()


# -------------------------------------------------
# helpers: choose nearest nodes to target
# -------------------------------------------------
def choose_nearest_nodes(x_all, y_all, x_target, k):
    # sort by distance to target
    triples = [(abs(x - x_target), i, x) for i, x in enumerate(x_all)]
    triples.sort(key=lambda t: t[0])
    chosen = triples[:k]
    # keep in x-order
    chosen.sort(key=lambda t: t[2])
    idxs = [t[1] for t in chosen]
    x_nodes = [x_all[i] for i in idxs]
    y_nodes = [y_all[i] for i in idxs]
    return x_nodes, y_nodes


# -------------------------------------------------
# Lagrange
# -------------------------------------------------
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


# -------------------------------------------------
# Newton divided difference
# -------------------------------------------------
def divided_difference_table(x_nodes, y_nodes):
    n = len(x_nodes)
    table = [y_nodes[:]]
    for level in range(1, n):
        prev = table[level - 1]
        col = []
        for i in range(n - level):
            num = prev[i + 1] - prev[i]
            den = x_nodes[i + level] - x_nodes[i]
            col.append(num / den)
        table.append(col)
    return table


def newton_coeffs_from_table(table):
    return [col[0] for col in table]


def newton_evaluate(x_nodes, coeffs, x):
    val = coeffs[0]
    prod = 1.0
    for i in range(1, len(coeffs)):
        prod *= (x - x_nodes[i - 1])
        val += coeffs[i] * prod
    return val


# -------------------------------------------------
# curve generators
# -------------------------------------------------
def lagrange_curve(x_nodes, y_nodes, xmin=0, xmax=100, num=200):
    xs = []
    ys = []
    step = (xmax - xmin) / (num - 1)
    for k in range(num):
        x = xmin + k * step
        y = lagrange_value(x_nodes, y_nodes, x)
        xs.append(x)
        ys.append(y)
    return xs, ys


def newton_curve(x_nodes, coeffs, xmin=0, xmax=100, num=200):
    xs = []
    ys = []
    step = (xmax - xmin) / (num - 1)
    for k in range(num):
        x = xmin + k * step
        y = newton_evaluate(x_nodes, coeffs, x)
        xs.append(x)
        ys.append(y)
    return xs, ys


# -------------------------------------------------
# plotting
# -------------------------------------------------
def plot_full_curves(lagrange_xy, newton_xy, x_data, y_data):
    """plot only the FULL-degree curves on [0,100] as lab asked"""
    plt.figure()
    plt.scatter(x_data, y_data, label="sensor data", zorder=5)
    xl, yl = lagrange_xy
    xn, yn = newton_xy
    plt.plot(xl, yl, label="Lagrange (full degree)")
    plt.plot(xn, yn, label="Newton (full degree)", linestyle="--")
    plt.xlabel("Distance (km)")
    plt.ylabel("Temperature (°C)")
    plt.title("Interpolated Temperature Along Highway (full degree)")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_local_low_degree(low_deg_polys):
    """
    show 2nd, 3rd, 4th-degree polynomials only over the span of
    the nodes they used, so they don't look weird
    """
    plt.figure()
    for label, x_nodes, fn in low_deg_polys:
        xmin = min(x_nodes)
        xmax = max(x_nodes)
        xs = [xmin + (xmax - xmin) * t / 200 for t in range(201)]
        ys = [fn(x) for x in xs]
        plt.plot(xs, ys, label=label)
    # we can also show sensor points for context
    plt.scatter(X_ALL, T_ALL, label="sensor data", zorder=5)
    plt.xlabel("Distance (km)")
    plt.ylabel("Temperature (°C)")
    plt.title("Local Interpolants (2nd, 3rd, 4th degree)")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_convergence(degrees, L_vals, N_vals):
    plt.figure()
    plt.plot(degrees, L_vals, marker="o", label="Lagrange T(45)")
    plt.plot(degrees, N_vals, marker="s", label="Newton T(45)")
    plt.xlabel("Polynomial degree")
    plt.ylabel("Temperature at 45 km (°C)")
    plt.title("Convergence of Interpolation at x = 45 km")
    plt.grid(True)
    plt.legend()
    plt.show()


# -------------------------------------------------
# main logic for lab report
# -------------------------------------------------
def main():
    print_data_table()

    # degrees: 2nd, 3rd, 4th, full (8th)
    node_counts = [3, 4, 5, len(X_ALL)]
    lagrange_results = []  # (degree, value_at_45)
    newton_results = []    # (degree, value_at_45)
    local_polys = []       # for nice local plot

    print("=== Lagrange interpolation at x = 45 km ===")
    prev_val = None
    for k in node_counts:
        x_nodes, y_nodes = choose_nearest_nodes(X_ALL, T_ALL, X_TARGET, k)
        val = lagrange_value(x_nodes, y_nodes, X_TARGET)
        degree = k - 1
        if prev_val is None:
            delta = None
        else:
            delta = val - prev_val
        print(f"P_{degree}(45) using nodes {x_nodes} -> {val:.8f} °C", end="")
        if delta is not None:
            print(f"   Δ = {delta:.8f}")
        else:
            print("   Δ = (first)")
        prev_val = val
        lagrange_results.append((degree, val))

        # save local lagrange function for plotting (only for low degrees)
        if k != len(X_ALL):
            def make_lag_fn(xn=x_nodes, yn=y_nodes):
                return lambda xx: lagrange_value(xn, yn, xx)
            local_polys.append((f"Lagrange P_{degree}", x_nodes, make_lag_fn()))

    print()

    print("=== Newton’s divided-difference interpolation at x = 45 km ===")
    prev_val = None
    for k in node_counts:
        x_nodes, y_nodes = choose_nearest_nodes(X_ALL, T_ALL, X_TARGET, k)
        table = divided_difference_table(x_nodes, y_nodes)
        coeffs = newton_coeffs_from_table(table)
        val = newton_evaluate(x_nodes, coeffs, X_TARGET)
        degree = k - 1
        if prev_val is None:
            delta = None
        else:
            delta = abs(val - prev_val)
        print(f"N_{degree}(45) using nodes {x_nodes} -> {val:.8f} °C", end="")
        if delta is not None:
            print(f"   Δ = {delta:.8f}")
        else:
            print("   Δ = (first)")
        prev_val = val
        newton_results.append((degree, val))

        # save local newton function too (only for low degrees)
        if k != len(X_ALL):
            def make_newt_fn(xn=x_nodes, cf=coeffs):
                return lambda xx: newton_evaluate(xn, cf, xx)
            local_polys.append((f"Newton N_{degree}", x_nodes, make_newt_fn()))

    print()

    # full-degree curves for plotting
    # Lagrange full
    x_full_lag, y_full_lag = lagrange_curve(X_ALL, T_ALL, xmin=0, xmax=100, num=200)

    # Newton full
    full_table = divided_difference_table(X_ALL, T_ALL)
    full_coeffs = newton_coeffs_from_table(full_table)
    x_full_newt, y_full_newt = newton_curve(X_ALL, full_coeffs, xmin=0, xmax=100, num=200)

    # plot interpolated vs data (full-degree only)
    plot_full_curves((x_full_lag, y_full_lag), (x_full_newt, y_full_newt), X_ALL, T_ALL)

    # plot local low-degree polynomials on their own spans (to avoid weird graph)
    plot_local_low_degree(local_polys)

    # convergence plot
    degrees = [d for (d, _) in lagrange_results]
    L_vals = [v for (_, v) in lagrange_results]
    N_vals = [v for (_, v) in newton_results]
    plot_convergence(degrees, L_vals, N_vals)


if __name__ == "__main__":
    main()
