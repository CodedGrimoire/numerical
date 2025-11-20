from typing import Callable, List, Tuple
import matplotlib.pyplot as plt


def heun_method(
    f: Callable[[float, float], float],
    x0: float,
    y0: float,
    h: float,
    x_end: float,
) -> Tuple[List[float], List[float]]:
    xs = [x0]
    ys = [y0]
    x, y = x0, y0

    while x < x_end - 1e-12:
        k1 = f(x, y)
        y_pred = y + h * k1
        k2 = f(x + h, y_pred)
        y = y + (h / 2.0) * (k1 + k2)
        x = x + h
        xs.append(x)
        ys.append(y)

    return xs, ys


def exact_solution(x: float) -> float:
    return -0.5 * x**4 + 4 * x**3 - 10 * x**2 + 8.5 * x + 1.0


def example_heun_with_error():
    def f(x, y):
        return -2 * x**3 + 12 * x**2 - 20 * x + 8.5

    x0, y0 = 0.0, 1.0
    h = 0.5
    x_end = 4.0

    xs, ys = heun_method(f, x0, y0, h, x_end)

    print("Heun's Method (with error)")
    print(f"{'step':>4} {'x':>6} {'y_num':>12} {'y_exact':>12} {'error':>12}")
    print("-" * 50)
    for i, (x, y_num) in enumerate(zip(xs, ys)):
        y_ex = exact_solution(x)
        err = abs(y_num - y_ex)
        print(f"{i:>4} {x:>6.2f} {y_num:>12.6f} {y_ex:>12.6f} {err:>12.6f}")

    # plot numeric vs exact
    y_exact_list = [exact_solution(x) for x in xs]

    plt.figure()
    plt.plot(xs, y_exact_list, label="Exact solution")
    plt.plot(xs, ys, marker="o", label="Heun")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Heun vs Exact")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    example_heun_with_error()
