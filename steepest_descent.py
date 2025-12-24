"""Steepest descent with exact line search for a quadratic form.

The quadratic is:
    q(x) = 1/2 (10 x1^2 - 18 x1 x2 + 10 x2^2) + 4 x1 - 15 x2 + 13
The global minimizer is x* = (5, 6) with q* = -22.

Running this module directly will execute steepest descent from a set of
predefined starting points and print the convergence factors based on
(q^{k+1} - q*) / (q^k - q*).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, List, Sequence, Tuple

# Quadratic data
A = ((10.0, -9.0), (-9.0, 10.0))  # symmetric Hessian
B = (4.0, -15.0)
C = 13.0

X_STAR = (5.0, 6.0)
Q_STAR = -22.0

# Ratios are unreliable when |q - q*| is below this gap because of rounding
MIN_GAP = 1e-10


@dataclass
class Iteration:
    x: Tuple[float, float]
    q: float
    grad_norm: float


def dot(u: Tuple[float, float], v: Tuple[float, float]) -> float:
    return u[0] * v[0] + u[1] * v[1]


def matvec(matrix: Tuple[Tuple[float, float], Tuple[float, float]], vec: Tuple[float, float]) -> Tuple[float, float]:
    return (
        matrix[0][0] * vec[0] + matrix[0][1] * vec[1],
        matrix[1][0] * vec[0] + matrix[1][1] * vec[1],
    )


def add(u: Tuple[float, float], v: Tuple[float, float]) -> Tuple[float, float]:
    return (u[0] + v[0], u[1] + v[1])


def scale(scalar: float, vec: Tuple[float, float]) -> Tuple[float, float]:
    return (scalar * vec[0], scalar * vec[1])


def norm(vec: Tuple[float, float]) -> float:
    return sqrt(dot(vec, vec))


def quadratic_value(x: Tuple[float, float]) -> float:
    ax = matvec(A, x)
    return 0.5 * dot(x, ax) + dot(B, x) + C


def quadratic_gradient(x: Tuple[float, float]) -> Tuple[float, float]:
    return add(matvec(A, x), B)


def exact_line_search_step(gradient: Tuple[float, float]) -> float:
    numerator = dot(gradient, gradient)
    ag = matvec(A, gradient)
    denominator = dot(gradient, ag)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def steepest_descent(
    x0: Sequence[float], *, tol: float = 1e-10, max_iter: int = 10_000
) -> List[Iteration]:
    x = (float(x0[0]), float(x0[1]))
    history: List[Iteration] = []

    for _ in range(max_iter):
        g = quadratic_gradient(x)
        q_val = quadratic_value(x)
        grad_norm = norm(g)
        history.append(Iteration(x=x, q=q_val, grad_norm=grad_norm))

        if grad_norm <= tol:
            break

        step = exact_line_search_step(g)
        x = add(x, scale(-step, g))

    return history


def ratio_sequence(q_values: Iterable[float], q_star: float, *, min_gap: float = MIN_GAP) -> List[float]:
    q_list = list(q_values)
    ratios: List[float] = []
    for k in range(len(q_list) - 1):
        numerator = q_list[k + 1] - q_star
        denominator = q_list[k] - q_star
        if abs(denominator) < min_gap or abs(numerator) < min_gap:
            continue
        ratios.append(numerator / denominator)
    return ratios


def eigenvalues_2x2_symmetric(matrix: Tuple[Tuple[float, float], Tuple[float, float]]) -> Tuple[float, float]:
    a, b = matrix[0]
    _, d = matrix[1]
    trace = a + d
    det = a * d - b * b
    term = sqrt(trace * trace - 4 * det)
    lambda1 = 0.5 * (trace + term)
    lambda2 = 0.5 * (trace - term)
    return (lambda1, lambda2)


def theoretical_linear_rate(matrix: Tuple[Tuple[float, float], Tuple[float, float]]) -> float:
    eigen1, eigen2 = eigenvalues_2x2_symmetric(matrix)
    lambda_max = max(eigen1, eigen2)
    lambda_min = min(eigen1, eigen2)
    condition_number = lambda_max / lambda_min
    return ((condition_number - 1) / (condition_number + 1)) ** 2


def format_vector(vec: Tuple[float, float]) -> str:
    return f"({vec[0]:.6f}, {vec[1]:.6f})"


def report_for_start(x0: Sequence[float]) -> str:
    history = steepest_descent(x0)
    q_vals = [item.q for item in history]
    ratios = ratio_sequence(q_vals, Q_STAR)
    tail = ratios[-5:] if len(ratios) >= 5 else ratios
    limsup_estimate = max(tail) if tail else 0.0
    final_x = history[-1].x
    final_q = history[-1].q
    iters = len(history) - 1  # number of steps taken

    lines = [
        f"x0 = {x0}",
        f"  iterations: {iters}",
        f"  final x: {format_vector(final_x)}",
        f"  q(final): {final_q:.10f}",
        f"  tail ratios (|q - q*| >= {MIN_GAP:g}): {tail}",
        f"  estimated limsup: {limsup_estimate:.6f}",
    ]
    return "\n".join(lines)


def main() -> None:
    initial_points = [
        (0.0, 0.0),
        (-0.4, 0.0),
        (10.0, 0.0),
        (11.0, 0.0),
    ]

    print("Steepest descent with exact line search for q(x)")
    print(f"Theoretical linear rate: {theoretical_linear_rate(A):.6f}\n")

    for x0 in initial_points:
        print(report_for_start(x0))
        print("-")


if __name__ == "__main__":
    main()
