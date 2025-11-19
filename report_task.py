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
    triples = [(abs(x - x_target), i, x) for i, x in enumerate(x_all)]
    triples.sort(key=lambda t: t[0])
    chosen = triples[:k]
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
# Convert Newton → Monomial (for full polynomial print)
# -------------------------------------------------
def newton_to_monomial(x_nodes, coeffs):
    n = len(coeffs)
    poly = [1.0]
    result = [0.0] * n
    for k in range(n):
        for i, c in enumerate(poly):
            result[i] += coeffs[k] * c
        if k < n - 1:
            xk = x_nodes[k]
            new_poly = [0.0] * (len(poly) + 1)
            for i, c in enumerate(poly):
                new_poly[i + 1] += c
                new_poly[i] -= xk * c
            poly = new_poly
    return result


def print_polynomial(coeffs):
    print("\nFINAL FULL-DEGREE POLYNOMIAL (degree 8):\n")
    terms = []
    for i, c in enumerate(coeffs):
        terms.append(f"{c:.10e} * x^{i}")
    print("P(x) = " + " + ".join(terms))
    print()


# -------------------------------------------------
# curve generators
# -------------------------------------------------
def lagrange_curve(x_nodes, y_nodes, xmin=0, xmax=100, num=200):
    xs, ys = [], []
    step = (xmax - xmin) / (num - 1)
    for k in range(num):
        x = xmin + k * step
        ys.append(lagrange_value(x_nodes, y_nodes, x))
        xs.append(x)
    return xs, ys


def newton_curve(x_nodes, coeffs, xmin=0, xmax=100, num=200):
    xs, ys = [], []
    step = (xmax - xmin) / (num - 1)
    for k in range(num):
        x = xmin + k * step
        ys.append(newton_evaluate(x_nodes, coeffs, x))
        xs.append(x)
    return xs, ys


# -------------------------------------------------
# NEW: Plot ALL requested degrees in one graph
# -------------------------------------------------
def plot_all_interpolations(node_counts, X_ALL, T_ALL):
    plt.figure(figsize=(10,6))

    plt.scatter(X_ALL, T_ALL, color="black", s=50, label="Sensor Data")

    colors = ["blue", "green", "orange", "purple"]

    for idx, k in enumerate(node_counts):
        deg = k - 1
        x_nodes, y_nodes = choose_nearest_nodes(X_ALL, T_ALL, X_TARGET, k)

        # --- Lagrange ---
        xs_L, ys_L = lagrange_curve(x_nodes, y_nodes)
        plt.plot(xs_L, ys_L, color=colors[idx],
                 label=f"Lagrange Degree {deg}", linewidth=2)

        # --- Newton ---
        table = divided_difference_table(x_nodes, y_nodes)
        coeffs = newton_coeffs_from_table(table)
        xs_N, ys_N = newton_curve(x_nodes, coeffs)
        plt.plot(xs_N, ys_N, linestyle="--",
                 color=colors[idx],
                 label=f"Newton Degree {deg}", linewidth=2)

    plt.xlabel("Distance (km)")
    plt.ylabel("Temperature (°C)")
    plt.title("Interpolating Polynomials of Degree 2, 3, 4, and 8 (Lagrange & Newton)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -------------------------------------------------
# Convergence plots
# -------------------------------------------------
def plot_convergence(degrees, L_vals, N_vals):
    plt.figure(figsize=(9,5))
    plt.plot(degrees, L_vals, marker="o", color="blue", label="Lagrange T(45)")
    shifted = [d + 0.05 for d in degrees]
    plt.plot(shifted, N_vals, marker="s", linestyle="--", color="red", label="Newton T(45)")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Temperature at 45 km (°C)")
    plt.title("Convergence Curve")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_delta_convergence(degrees, L_deltas, N_deltas):
    plt.figure(figsize=(9,5))
    plt.plot(degrees[1:], L_deltas[1:], marker="o", color="blue",
             label="Δₖ Lagrange")
    plt.plot(degrees[1:], N_deltas[1:], marker="s", linestyle="--",
             color="red", label="Δₖ Newton")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Δₖ = Pₖ(45) − Pₖ₋₁(45)")
    plt.title("Δₖ Convergence (Successive Differences)")
    plt.grid(True)
    plt.legend()
    plt.show()


# -------------------------------------------------
# main logic
# -------------------------------------------------
def main():
    print_data_table()

    # Required: 3,4,5,9 nodes → degrees 2,3,4,8
    node_counts = [3, 4, 5, len(X_ALL)]

    lagrange_results = []
    newton_results = []
    lagrange_deltas = []
    newton_deltas = []

    print("=== Lagrange interpolation at x = 45 km ===")
    prev_val = None
    for k in node_counts:
        x_nodes, y_nodes = choose_nearest_nodes(X_ALL, T_ALL, X_TARGET, k)
        val = lagrange_value(x_nodes, y_nodes, X_TARGET)
        if prev_val is None:
            lagrange_deltas.append(None)
        else:
            lagrange_deltas.append(val - prev_val)
        prev_val = val
        degree = k - 1
        print(f"Lagrange deg {degree}: {val:.8f}")
        lagrange_results.append((degree, val))

    print("\n=== Newton interpolation at x = 45 km ===")
    prev_val = None
    for k in node_counts:
        x_nodes, y_nodes = choose_nearest_nodes(X_ALL, T_ALL, X_TARGET, k)
        table = divided_difference_table(x_nodes, y_nodes)
        coeffs = newton_coeffs_from_table(table)
        val = newton_evaluate(x_nodes, coeffs, X_TARGET)
        if prev_val is None:
            newton_deltas.append(None)
        else:
            newton_deltas.append(val - prev_val)
        prev_val = val
        degree = k - 1
        print(f"Newton deg {degree}: {val:.8f}")
        newton_results.append((degree, val))

    # Full polynomial print
    full_table = divided_difference_table(X_ALL, T_ALL)
    full_coeffs = newton_coeffs_from_table(full_table)
    monomial = newton_to_monomial(X_ALL, full_coeffs)
    print_polynomial(monomial)

    # ---- PLOT ALL DEGREES IN ONE GRAPH ----
    plot_all_interpolations(node_counts, X_ALL, T_ALL)

    # ---- Convergence ----
    degrees = [d for (d, _) in lagrange_results]
    L_vals = [v for (_, v) in lagrange_results]
    N_vals = [v for (_, v) in newton_results]

    plot_convergence(degrees, L_vals, N_vals)
    plot_delta_convergence(degrees, lagrange_deltas, newton_deltas)


if __name__ == "__main__":
    main()
