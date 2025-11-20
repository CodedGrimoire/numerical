from typing import Callable, List, Tuple, Dict
import matplotlib.pyplot as plt


def heun_solve_with_logs(
    f: Callable[[float, float], float],
    exact: Callable[[float], float],
    x0: float,
    y0: float,
    h: float,
    x_end: float,
) -> Tuple[List[float], List[float], List[Dict]]:
    """
    Heun's method (Improved Euler / RK2) for y' = f(x, y) with detailed per-step logs.

    Returns:
        xs, ys : list of x and approximate y at each node
        logs   : list of dicts, each describing one step:
                 {
                     "step": i,
                     "x_i": x_i,
                     "y_i": y_i,
                     "k1": f(x_i, y_i),
                     "y_predict": y_i + h * k1,
                     "k2": f(x_{i+1}, y_predict),
                     "x_next": x_{i+1},
                     "y_next": y_{i+1},
                     "y_exact_next": y(x_{i+1}),
                     "error_next": |y_{i+1} - y_exact(x_{i+1})|
                 }
    """
    xs = [x0]
    ys = [y0]
    logs: List[Dict] = []

    x = x0
    y = y0
    step = 0

    while x < x_end - 1e-12:
        k1 = f(x, y)
        x_next = x + h
        y_predict = y + h * k1
        k2 = f(x_next, y_predict)

        y_next = y + (h / 2.0) * (k1 + k2)

        y_exact_next = exact(x_next)
        error_next = abs(y_next - y_exact_next)

        logs.append(
            {
                "step": step,
                "x_i": x,
                "y_i": y,
                "k1": k1,
                "y_predict": y_predict,
                "k2": k2,
                "x_next": x_next,
                "y_next": y_next,
                "y_exact_next": y_exact_next,
                "error_next": error_next,
            }
        )

        xs.append(x_next)
        ys.append(y_next)

        x, y = x_next, y_next
        step += 1

    return xs, ys, logs


def exact_solution(x: float) -> float:
    """
    Exact solution:
        y(x) = -0.5 x^4 + 4 x^3 - 10 x^2 + 8.5 x + 1
    """
    return -0.5 * x**4 + 4.0 * x**3 - 10.0 * x**2 + 8.5 * x + 1.0


def print_step_table(logs: List[Dict]) -> None:
    """
    Print detailed step-by-step Heun table.
    """
    print("Heun's Method – Detailed Step Table")
    print(
        f"{'step':>4} {'x_i':>8} {'y_i':>12} "
        f"{'k1':>12} {'y_predict':>12} {'k2':>12} "
        f"{'x_{i+1}':>8} {'y_{i+1}':>12} {'y_exact(x_{i+1})':>18} {'error':>12}"
    )
    print("-" * 120)

    for row in logs:
        print(
            f"{row['step']:>4} "
            f"{row['x_i']:>8.3f} "
            f"{row['y_i']:>12.6f} "
            f"{row['k1']:>12.6f} "
            f"{row['y_predict']:>12.6f} "
            f"{row['k2']:>12.6f} "
            f"{row['x_next']:>8.3f} "
            f"{row['y_next']:>12.6f} "
            f"{row['y_exact_next']:>18.6f} "
            f"{row['error_next']:>12.6f}"
        )

    max_err = max(r["error_next"] for r in logs)
    print("-" * 120)
    print(f"Max error (at nodes x_{'{i+1}'}) = {max_err:.6e}")


def plot_heun_vs_exact(xs: List[float], ys: List[float]) -> None:
    """
    Plot Heun approximation vs exact solution.
    """
    y_exact_list = [exact_solution(x) for x in xs]

    plt.figure()
    plt.plot(xs, y_exact_list, label="Exact solution")
    plt.plot(xs, ys, marker="o", label="Heun approximation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Heun's Method vs Exact Solution")
    plt.grid(True)
    plt.legend()
    plt.show()


def example():
    """
    Example IVP:
        y' = -2x^3 + 12x^2 - 20x + 8.5,  y(0) = 1
        on [0, 4] with h = 0.5
    """

    def f(x: float, y: float) -> float:
        return -2 * x**3 + 12 * x**2 - 20 * x + 8.5

    x0 = 0.0
    y0 = 1.0
    h = 0.5
    x_end = 4.0

    xs, ys, logs = heun_solve_with_logs(f, exact_solution, x0, y0, h, x_end)

    # 1) full per-step table
    print_step_table(logs)

    # 2) quick summary table (optional)
    print("\nNode Values (x, y_Heun, y_exact, error):")
    print(f"{'x':>8} {'y_Heun':>12} {'y_exact':>12} {'error':>12}")
    print("-" * 50)
    for x, y_num in zip(xs, ys):
        y_ex = exact_solution(x)
        err = abs(y_num - y_ex)
        print(f"{x:>8.3f} {y_num:>12.6f} {y_ex:>12.6f} {err:>12.6f}")

    # 3) plot
    plot_heun_vs_exact(xs, ys)


if __name__ == "__main__":
    example()
