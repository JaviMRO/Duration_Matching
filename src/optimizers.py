"""
Métodos de optimización sin restricciones para el problema de duration matching:

    min_w  f(w) = (D^T w - D_L)^2 + lam * ||w - w0||^2

- Descenso por gradiente:  w_{k+1} = w_k - alpha_k grad f(w_k)
    * paso fijo alpha = 1/L, con L = lambda_max(Hess f) (constante de Lipschitz del gradiente),
    * o paso por backtracking con la condición de Armijo.
- Newton:  w_{k+1} = w_k + alpha_k p_k,  con  Hess f(w_k) p_k = -grad f(w_k),
    con alpha_k = 1 (Newton puro) o elegido por backtracking de Armijo.

Como f es cuadrática estrictamente convexa, el mínimo tiene solución cerrada
    (D D^T + lam I) w* = D_L D + lam w0,
que se usa como referencia para medir el error de cada método.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from objective import ALMProblem, load_problem


@dataclass
class OptResult:
    """Resultado de un método de optimización."""

    method: str
    w: np.ndarray
    f: float
    grad_norm: float
    n_iter: int
    converged: bool
    time_s: float
    f_hist: list[float] = field(default_factory=list)          # f(w_k), k = 0..n_iter
    grad_norm_hist: list[float] = field(default_factory=list)  # ||grad f(w_k)||
    step_hist: list[float] = field(default_factory=list)       # alpha_k usado


def armijo_backtracking(
    f: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    w: np.ndarray,
    fw: float,
    g: np.ndarray,
    p: np.ndarray,
    alpha0: float = 1.0,
    rho: float = 0.5,
    c: float = 1e-4,
    max_halvings: int = 60,
) -> float:
    """
    Backtracking: devuelve el primer alpha = alpha0 * rho^j que cumple la condición de Armijo
        phi(alpha) <= phi(0) + c * alpha * phi'(0),   phi(alpha) = f(w + alpha p).

    Cerca del óptimo, phi(alpha) - phi(0) cae por debajo de la precisión de coma flotante de
    f(w) (eps_f = 10 * eps_maq * |f(w)|) y la comparación anterior la decide el redondeo:
    alpha colapsa a 0 y el método se estanca. En ese caso se usa la condición de Armijo
    aproximada de Hager-Zhang (2005), basada en derivadas direccionales (que sí son precisas):
        phi'(alpha) <= (1 - 2c) * |phi'(0)|,    phi'(alpha) = grad f(w + alpha p)^T p.
    Para una f cuadrática ambas condiciones son equivalentes.
    Requiere que p sea dirección de descenso (phi'(0) = grad f(w)^T p < 0).
    """
    slope = float(g @ p)
    if slope >= 0:
        raise ValueError("p no es dirección de descenso")
    eps_f = 10 * np.finfo(float).eps * abs(fw)
    alpha = alpha0
    for _ in range(max_halvings):
        w_new = w + alpha * p
        f_new = f(w_new)
        if abs(f_new - fw) > eps_f:
            if f_new <= fw + c * alpha * slope:
                return alpha
        elif float(grad(w_new) @ p) <= (1 - 2 * c) * abs(slope):
            return alpha
        alpha *= rho
    return alpha


def gradient_descent(
    problem: ALMProblem,
    w_init: np.ndarray,
    step: str = "fixed",
    tol: float = 1e-8,
    max_iter: int = 500_000,
) -> OptResult:
    """
    Descenso por gradiente: w_{k+1} = w_k - alpha_k grad f(w_k).

    step = "fixed":  alpha_k = 1/L, con L = lambda_max(2 D D^T + 2 lam I) = 2||D||^2 + 2 lam.
    step = "armijo": alpha_k por backtracking de Armijo partiendo de alpha0 = 1.
    Criterio de parada: ||grad f(w_k)|| < tol.
    La convergencia es lineal con factor (1 - mu/L) = (1 - 1/kappa) por iteración, por lo que
    con kappa ~ 8e3 puede requerir del orden de 1e5 iteraciones (de ahí max_iter elevado).
    """
    if step not in ("fixed", "armijo"):
        raise ValueError("step debe ser 'fixed' o 'armijo'")
    L = float(np.linalg.eigvalsh(problem.hess(w_init))[-1])

    t0 = time.perf_counter()
    w = np.asarray(w_init, dtype=float).copy()
    fw, g = problem.f(w), problem.grad(w)
    f_hist, gn_hist, step_hist = [fw], [float(np.linalg.norm(g))], []

    k = 0
    while gn_hist[-1] >= tol and k < max_iter:
        p = -g
        if step == "fixed":
            alpha = 1.0 / L
        else:
            alpha = armijo_backtracking(problem.f, problem.grad, w, fw, g, p)
        w = w + alpha * p
        fw, g = problem.f(w), problem.grad(w)
        k += 1
        f_hist.append(fw)
        gn_hist.append(float(np.linalg.norm(g)))
        step_hist.append(alpha)

    return OptResult(
        method=f"Gradiente ({'1/L' if step == 'fixed' else 'Armijo'})",
        w=w, f=fw, grad_norm=gn_hist[-1], n_iter=k, converged=gn_hist[-1] < tol,
        time_s=time.perf_counter() - t0,
        f_hist=f_hist, grad_norm_hist=gn_hist, step_hist=step_hist,
    )


def newton(
    problem: ALMProblem,
    w_init: np.ndarray,
    line_search: bool = False,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> OptResult:
    """
    Método de Newton: resuelve Hess f(w_k) p_k = -grad f(w_k) y actualiza w_{k+1} = w_k + alpha_k p_k.

    line_search = False: Newton puro (alpha_k = 1).
    line_search = True:  alpha_k por backtracking de Armijo partiendo de alpha0 = 1.
    Se resuelve el sistema lineal (np.linalg.solve) en lugar de invertir la Hessiana.
    Criterio de parada: ||grad f(w_k)|| < tol.
    """
    t0 = time.perf_counter()
    w = np.asarray(w_init, dtype=float).copy()
    fw, g = problem.f(w), problem.grad(w)
    f_hist, gn_hist, step_hist = [fw], [float(np.linalg.norm(g))], []

    k = 0
    while gn_hist[-1] >= tol and k < max_iter:
        p = np.linalg.solve(problem.hess(w), -g)
        alpha = armijo_backtracking(problem.f, problem.grad, w, fw, g, p) if line_search else 1.0
        w = w + alpha * p
        fw, g = problem.f(w), problem.grad(w)
        k += 1
        f_hist.append(fw)
        gn_hist.append(float(np.linalg.norm(g)))
        step_hist.append(alpha)

    return OptResult(
        method=f"Newton ({'Armijo' if line_search else 'puro'})",
        w=w, f=fw, grad_norm=gn_hist[-1], n_iter=k, converged=gn_hist[-1] < tol,
        time_s=time.perf_counter() - t0,
        f_hist=f_hist, grad_norm_hist=gn_hist, step_hist=step_hist,
    )


def closed_form_solution(problem: ALMProblem) -> np.ndarray:
    """Mínimo exacto: grad f(w*) = 0  <=>  (D D^T + lam I) w* = D_L D + lam w0."""
    A = np.outer(problem.D, problem.D) + problem.lam * np.eye(problem.n)
    b = problem.D_L * problem.D + problem.lam * problem.w0
    return np.linalg.solve(A, b)


def run_all(problem: ALMProblem, w_init: np.ndarray, tol: float = 1e-8) -> list[OptResult]:
    """Ejecuta los cuatro métodos desde el mismo punto inicial."""
    return [
        gradient_descent(problem, w_init, step="fixed", tol=tol),
        gradient_descent(problem, w_init, step="armijo", tol=tol),
        newton(problem, w_init, line_search=False, tol=tol),
        newton(problem, w_init, line_search=True, tol=tol),
    ]


def print_table(problem: ALMProblem, results: list[OptResult], w_star: np.ndarray) -> None:
    """Tabla comparativa: iteraciones, f final, gap, ||grad||, error frente a w* y tiempo."""
    header = (f"  {'Método':<20}{'iter':>8}{'f(w)':>14}{'gap (años)':>13}"
              f"{'||grad||':>11}{'||w-w*||':>11}{'tiempo (ms)':>13}  conv")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        print(f"  {r.method:<20}{r.n_iter:>8}{r.f:>14.8f}{problem.duration_gap(r.w):>+13.6f}"
              f"{r.grad_norm:>11.2e}{np.linalg.norm(r.w - w_star):>11.2e}"
              f"{1e3 * r.time_s:>13.2f}  {'sí' if r.converged else 'NO'}")


if __name__ == "__main__":
    prob = load_problem(lam=0.1)
    w_star = closed_form_solution(prob)
    eig = np.linalg.eigvalsh(prob.hess(prob.w0))

    print(f"Problema: n = {prob.n}, D_L = {prob.D_L}, lam = {prob.lam}")
    print(f"  L = lambda_max = {eig[-1]:.4f},  mu = lambda_min = {eig[0]:.4f},  "
          f"kappa = {eig[-1] / eig[0]:.1f}")
    print(f"  Solución cerrada: f(w*) = {prob.f(w_star):.8f}, "
          f"gap(w*) = {prob.duration_gap(w_star):+.6f} años")

    starts = {
        "w0 (cartera actual)": prob.w0,
        "uniforme (1/n)": np.full(prob.n, 1.0 / prob.n),
    }
    all_results: dict[str, list[OptResult]] = {}
    for name, w_init in starts.items():
        print(f"\nPunto inicial: {name}  ->  f = {prob.f(w_init):.6f}, "
              f"gap = {prob.duration_gap(w_init):+.4f} años")
        results = run_all(prob, w_init)
        print_table(prob, results, w_star)
        all_results[name] = results
        for r in results:
            assert r.converged, f"{r.method} no convergió desde {name}"
            assert np.linalg.norm(r.w - w_star) < 1e-6, f"{r.method} no llegó a w*"

    # Todas las soluciones coinciden con w*; se muestra w* frente a la cartera actual.
    print("\nCartera óptima (sin restricciones) frente a la actual:")
    print(f"  {'id':<5}{'D_i':>8}{'w0':>10}{'w*':>10}{'w*-w0':>10}")
    for i, id_ in enumerate(prob.ids):
        print(f"  {id_:<5}{prob.D[i]:>8.3f}{prob.w0[i]:>10.4f}{w_star[i]:>10.4f}"
              f"{w_star[i] - prob.w0[i]:>+10.4f}")
    print(f"  sum(w*) = {w_star.sum():.4f}   min(w*) = {w_star.min():+.4f}   "
          f"max(w*) = {w_star.max():.4f}  (u_max = {prob.u_max.max():.2f})")
    n_viol = int(np.sum(w_star < 0) + np.sum(w_star > prob.u_max))
    print(f"  Instrumentos fuera de [0, u_max]: {n_viol}  "
          "(el problema aún no impone restricciones)")
