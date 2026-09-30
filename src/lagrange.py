"""
Problema con restricción de presupuesto (igualdad) resuelto por Lagrange:

    min_w  f(w) = (D^T w - D_L)^2 + lam * ||w - w0||^2     s.a.  1^T w = 1

Lagrangiana:  L(w, nu) = f(w) + nu * (1^T w - 1).
Condiciones KKT (f cuadrática => sistema lineal):
    grad_w L = H w - 2 (D_L D + lam w0) + nu 1 = 0,     H = 2 (D D^T + lam I)
    1^T w = 1
es decir
    [ H   1 ] [ w  ]   [ 2 (D_L D + lam w0) ]
    [ 1^T 0 ] [ nu ] = [ 1                  ].
La matriz KKT es no singular porque H es definida positiva y el vector 1 no es nulo.

Interpretación económica del multiplicador nu
---------------------------------------------
Si la restricción de presupuesto fuera 1^T w = b, el valor óptimo f*(b) cumple
(teorema de la envolvente)
    d f*(b) / d b = -nu.
Por tanto -nu mide cuánto cambia la función objetivo por cada unidad adicional de
presupuesto (relajar marginalmente la restricción de 1 a 1 + eps cambia f* en -nu * eps):
    * nu < 0: aumentar el presupuesto EMPEORA f (más peso invertido => más duración, y el
      problema quería invertir menos para bajar D^T w); relajar la restricción hacia abajo
      (b < 1) mejoraría f en |nu| por unidad.
    * nu > 0: aumentar el presupuesto mejoraría f en nu por unidad.
Es el "precio sombra" de la restricción de que la cartera esté totalmente invertida.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from objective import ALMProblem, load_problem
from optimizers import closed_form_solution


@dataclass
class LagrangeResult:
    """Solución KKT del problema con restricción de presupuesto."""

    w: np.ndarray
    nu: float
    gap: float
    f: float


def solve_kkt(problem: ALMProblem, budget: float = 1.0) -> LagrangeResult:
    """
    Resuelve el sistema KKT
        [ H 1; 1^T 0 ] [w; nu] = [ 2 (D_L D + lam w0); budget ],  H = 2 (D D^T + lam I),
    con numpy.linalg.solve. budget = 1 es el problema original; otros valores se usan
    para verificar la interpretación de nu como sensibilidad d f*/d budget = -nu.
    """
    n = problem.n
    H = problem.hess(problem.w0)  # = 2 (D D^T + lam I), constante
    ones = np.ones(n)
    K = np.block([[H, ones[:, None]], [ones[None, :], np.zeros((1, 1))]])
    rhs = np.concatenate([2.0 * (problem.D_L * problem.D + problem.lam * problem.w0), [budget]])
    sol = np.linalg.solve(K, rhs)
    w, nu = sol[:n], float(sol[n])
    return LagrangeResult(w=w, nu=nu, gap=problem.duration_gap(w), f=problem.f(w))


def lagrangian_grad(problem: ALMProblem, w: np.ndarray, nu: float) -> np.ndarray:
    """Gradiente de la lagrangiana respecto a w: grad f(w) + nu * 1."""
    return problem.grad(w) + nu * np.ones(problem.n)


def solve_slsqp(problem: ALMProblem, w_init: np.ndarray) -> np.ndarray:
    """Resuelve el mismo problema con scipy.optimize.minimize (SLSQP, solo 1^T w = 1)."""
    res = minimize(
        problem.f,
        w_init,
        jac=problem.grad,
        method="SLSQP",
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones_like(w)}],
        options={"ftol": 1e-14, "maxiter": 1000},
    )
    if not res.success:
        raise RuntimeError(f"SLSQP falló: {res.message}")
    return res.x


def print_weights(problem: ALMProblem, w: np.ndarray, w_unc: np.ndarray) -> None:
    """Tabla de pesos: w0, óptimo sin restricciones y óptimo de Lagrange, con marcas."""
    print(f"  {'id':<5}{'D_i':>8}{'w0':>10}{'w_sinR':>10}{'w_Lagr':>10}  marca")
    for i, id_ in enumerate(problem.ids):
        mark = "NEGATIVO" if w[i] < 0 else ("> u_max" if w[i] > problem.u_max[i] else "")
        print(f"  {id_:<5}{problem.D[i]:>8.3f}{problem.w0[i]:>10.4f}{w_unc[i]:>10.4f}"
              f"{w[i]:>10.4f}  {mark}")
    n_neg = int(np.sum(w < 0))
    n_up = int(np.sum(w > problem.u_max))
    print(f"  sum(w) = {w.sum():.12f}   negativos: {n_neg}   por encima de u_max: {n_up}")


if __name__ == "__main__":
    prob = load_problem(lam=0.1)
    res = solve_kkt(prob)

    print(f"Lagrange (1^T w = 1), lam = {prob.lam}")
    print(f"  f(w*)       = {res.f:.8f}")
    print(f"  gap final   = {res.gap:+.6f} años   (inicial {prob.duration_gap(prob.w0):+.4f})")
    print(f"  nu          = {res.nu:+.8f}")

    # Verificaciones
    budget_err = abs(res.w.sum() - 1.0)
    grad_L = np.linalg.norm(lagrangian_grad(prob, res.w, res.nu))
    print("\nVerificaciones:")
    print(f"  |sum(w*) - 1|         = {budget_err:.2e}")
    print(f"  ||grad_w L(w*, nu)||  = {grad_L:.2e}")
    assert budget_err < 1e-10, "No se cumple la restricción de presupuesto"
    assert grad_L < 1e-8, "El gradiente de la lagrangiana no es ~0"

    # Sensibilidad: d f*/d b ≈ -nu (diferencias centrales en el presupuesto)
    eps = 1e-5
    df_db = (solve_kkt(prob, 1 + eps).f - solve_kkt(prob, 1 - eps).f) / (2 * eps)
    print(f"  d f*/d b (dif. finitas) = {df_db:+.8f}   vs  -nu = {-res.nu:+.8f}")
    assert abs(df_db + res.nu) < 1e-6 * max(1.0, abs(res.nu)), "Sensibilidad no coincide con -nu"

    # Validación con SLSQP
    w_slsqp = solve_slsqp(prob, prob.w0)
    diff = np.linalg.norm(w_slsqp - res.w)
    print(f"\nValidación SLSQP:  ||w_SLSQP - w_KKT|| = {diff:.2e},  "
          f"|f_SLSQP - f_KKT| = {abs(prob.f(w_slsqp) - res.f):.2e}")
    assert diff < 1e-6, "SLSQP y KKT no coinciden"
    print("  OK: coinciden")

    print("\nPesos óptimos:")
    print_weights(prob, res.w, closed_form_solution(prob))
