"""
Penalización exterior cuadrática para el problema con presupuesto y límites de caja:

    min_w  f(w) = (D^T w - D_L)^2 + lam ||w - w0||^2
    s.a.   1^T w = 1,   0 <= w_i <= u_i  (u = u_max)

Función penalizada (mu > 0):
    P_mu(w) = f(w) + (mu/2) [ (1^T w - 1)^2 + sum_i max(0, -w_i)^2 + sum_i max(0, w_i - u_i)^2 ]

Gradiente (exacto; max(0, t)^2 es C^1 con derivada 2 max(0, t)):
    grad P_mu(w) = grad f(w) + mu [ (1^T w - 1) 1 - max(0, -w) + max(0, w - u) ]

Hessiana generalizada (P_mu es C^1 y cuadrática a trozos):
    H_mu(w) = H_f + mu (1 1^T + diag(a)),   a_i = 1 si w_i < 0 o w_i > u_i, 0 en otro caso.

Bucle externo: mu = 1, 10, ..., 1e8, con warm start (cada subproblema arranca en la solución
del anterior; el primero en w0). Solver interno: Newton con Armijo de optimizers.newton.

Estimaciones de multiplicadores KKT (comparando grad P_mu = 0 con la estacionariedad de la
lagrangiana  grad f + nu 1 - lambda + eta = 0):
    nu       ≈ mu (1^T w - 1)             (presupuesto)
    lambda_i ≈ mu max(0, -w_i)            (w_i >= 0)
    eta_i    ≈ mu max(0, w_i - u_i)       (w_i <= u_i)
La solución penalizada es ligeramente infactible: la violación es O(multiplicador / mu).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from lagrange import solve_kkt
from objective import ALMProblem, load_problem
from optimizers import newton

MUS = [10.0**k for k in range(9)]  # 1, 10, ..., 1e8
INNER_TOL = 1e-10
INNER_MAX_ITER = 200
ACTIVE_TOL = 1e-6   # |w_i - límite| por debajo de esto => restricción de caja activa
FEAS_TOL = 1e-6     # tolerancia de factibilidad en la verificación KKT


@dataclass
class PenalizedProblem:
    """P_mu(w) con la misma interfaz (f, grad, hess) que ALMProblem, para reutilizar newton()."""

    base: ALMProblem
    mu: float

    def violations(self, w: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """(1^T w - 1, max(0, -w), max(0, w - u))."""
        return w.sum() - 1.0, np.maximum(0.0, -w), np.maximum(0.0, w - self.base.u_max)

    def f(self, w: np.ndarray) -> float:
        """P_mu(w) = f(w) + (mu/2) [ (1^T w - 1)^2 + ||max(0,-w)||^2 + ||max(0,w-u)||^2 ]."""
        r, lo, up = self.violations(w)
        return self.base.f(w) + 0.5 * self.mu * (r**2 + lo @ lo + up @ up)

    def grad(self, w: np.ndarray) -> np.ndarray:
        """grad P_mu(w) = grad f(w) + mu [ (1^T w - 1) 1 - max(0,-w) + max(0,w-u) ]."""
        r, lo, up = self.violations(w)
        return self.base.grad(w) + self.mu * (r - lo + up)

    def hess(self, w: np.ndarray) -> np.ndarray:
        """H_mu(w) = H_f + mu (1 1^T + diag(a)),  a_i = 1 si w_i < 0 o w_i > u_i."""
        a = ((w < 0) | (w > self.base.u_max)).astype(float)
        n = self.base.n
        return self.base.hess(w) + self.mu * (np.ones((n, n)) + np.diag(a))


@dataclass
class PenaltyStep:
    """Resumen de un subproblema del bucle externo."""

    mu: float
    w: np.ndarray
    n_inner: int
    converged: bool
    grad_norm: float
    max_violation: float
    gap: float
    budget: float
    cond: float
    f: float


def max_violation(problem: ALMProblem, w: np.ndarray) -> float:
    """max( |1^T w - 1|, max_i(-w_i), max_i(w_i - u_i), 0 )."""
    return float(max(abs(w.sum() - 1.0), np.max(-w), np.max(w - problem.u_max), 0.0))


def solve_penalty(problem: ALMProblem, mus: list[float] = MUS) -> list[PenaltyStep]:
    """Bucle externo de penalización con warm start; el solver interno es Newton + Armijo."""
    steps = []
    w = problem.w0.copy()
    for mu in mus:
        pp = PenalizedProblem(problem, mu)
        res = newton(pp, w, line_search=True, tol=INNER_TOL, max_iter=INNER_MAX_ITER)
        w = res.w
        eig = np.linalg.eigvalsh(pp.hess(w))
        steps.append(PenaltyStep(
            mu=mu, w=w.copy(), n_inner=res.n_iter, converged=res.converged,
            grad_norm=res.grad_norm, max_violation=max_violation(problem, w),
            gap=problem.duration_gap(w), budget=float(w.sum()),
            cond=float(eig[-1] / eig[0]), f=problem.f(w),
        ))
    return steps


def penalty_multipliers(problem: ALMProblem, w: np.ndarray, mu: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimaciones (nu, lambda, eta) = (mu (1^T w - 1), mu max(0,-w), mu max(0, w-u))."""
    return mu * (w.sum() - 1.0), mu * np.maximum(0.0, -w), mu * np.maximum(0.0, w - problem.u_max)


def active_sets(problem: ALMProblem, w: np.ndarray, tol: float = ACTIVE_TOL) -> tuple[np.ndarray, np.ndarray]:
    """Máscaras de restricciones de caja activas: (w_i ≈ 0, w_i ≈ u_i)."""
    return np.abs(w) <= tol, np.abs(w - problem.u_max) <= tol


def kkt_multipliers_ls(problem: ALMProblem, w: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float]:
    """
    Estimación independiente de los multiplicadores: fija el conjunto activo detectado en w y
    resuelve por mínimos cuadrados  grad f(w) + nu 1 - sum_{i en A0} lambda_i e_i
    + sum_{i en Au} eta_i e_i = 0.  Devuelve (nu, lambda, eta, residuo de estacionariedad).
    """
    at0, atu = active_sets(problem, w)
    n = problem.n
    cols = [np.ones(n)] + [-np.eye(n)[i] for i in np.where(at0)[0]] + [np.eye(n)[i] for i in np.where(atu)[0]]
    A = np.column_stack(cols)
    g = problem.grad(w)
    coef, *_ = np.linalg.lstsq(A, -g, rcond=None)
    lam_, eta = np.zeros(n), np.zeros(n)
    lam_[at0] = coef[1:1 + at0.sum()]
    eta[atu] = coef[1 + at0.sum():]
    resid = float(np.linalg.norm(g + A @ coef))
    return float(coef[0]), lam_, eta, resid


def solve_slsqp(problem: ALMProblem) -> np.ndarray:
    """Referencia: scipy SLSQP con 1^T w = 1 y bounds 0 <= w <= u_max, desde w0."""
    res = minimize(
        problem.f, problem.w0, jac=problem.grad, method="SLSQP",
        bounds=list(zip(np.zeros(problem.n), problem.u_max)),
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones_like(w)}],
        options={"ftol": 1e-15, "maxiter": 1000},
    )
    if not res.success:
        raise RuntimeError(f"SLSQP falló: {res.message}")
    return res.x


def turnover(problem: ALMProblem, w: np.ndarray) -> float:
    """Turnover: sum_i |w_i - w0_i|."""
    return float(np.abs(w - problem.w0).sum())


def print_outer_table(steps: list[PenaltyStep]) -> None:
    """Una fila por valor de mu."""
    print(f"  {'mu':>7}{'iter':>6}{'conv':>6}{'||grad P||':>12}{'viol. máx':>12}{'gap (años)':>13}"
          f"{'sum(w)':>15}{'kappa(H_mu)':>13}{'f(w)':>13}")
    for s in steps:
        print(f"  {s.mu:>7.0e}{s.n_inner:>6}{'sí' if s.converged else 'NO':>6}{s.grad_norm:>12.2e}"
              f"{s.max_violation:>12.2e}{s.gap:>+13.2e}{s.budget:>15.10f}{s.cond:>13.3e}{s.f:>13.8f}")


def mark(problem: ALMProblem, w: np.ndarray, i: int) -> str:
    """'= 0' / '= u_max' si w_i está en un límite (tolerancia ACTIVE_TOL), 'NEG'/'> u' si lo viola."""
    if abs(w[i]) <= ACTIVE_TOL:
        return "= 0"
    if abs(w[i] - problem.u_max[i]) <= ACTIVE_TOL:
        return "= u_max"
    if w[i] < 0:
        return "NEG"
    if w[i] > problem.u_max[i]:
        return "> u_max"
    return ""


def plot_penalty_convergence(steps: list[PenaltyStep], path) -> None:
    """Violación máxima y kappa(H_mu) frente a mu (ejes log-log)."""
    import matplotlib.pyplot as plt
    from plots import BLUE, INK

    mus = [s.mu for s in steps]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, y, title, ylabel in [
        (ax1, [s.max_violation for s in steps], "Violación máxima de restricciones", "violación máx."),
        (ax2, [s.cond for s in steps], "Número de condición de la Hessiana", r"$\kappa(H_\mu)$"),
    ]:
        ax.plot(mus, y, color=BLUE, marker="o", markersize=8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\mu$ (escala log)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        for x, v in [(mus[0], y[0]), (mus[-1], y[-1])]:
            ax.annotate(f"{v:.1e}", (x, v), xytext=(6, 4), textcoords="offset points",
                        fontsize=8, color=INK)
    fig.suptitle("Penalización exterior cuadrática", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_weights_v2(problem: ALMProblem, w_lagr: np.ndarray, w_pen: np.ndarray, path) -> None:
    """Barras agrupadas: w0, sin restricciones, Lagrange y penalización (cuatro series)."""
    import matplotlib.pyplot as plt
    from optimizers import closed_form_solution
    from plots import AQUA, BLUE, INK, MUTED, ORANGE

    YELLOW = "#eda100"  # slot 4 de la paleta categórica validada
    series = [
        (problem.w0, BLUE, "Cartera actual $w_0$", None),
        (closed_form_solution(problem), ORANGE, "Óptimo sin restricciones", None),
        (w_lagr, AQUA, r"Lagrange ($\mathbf{1}^\top w = 1$)", "////"),
        (w_pen, YELLOW, r"Penalización ($\mathbf{1}^\top w = 1$, $0 \leq w \leq u$)", "\\\\\\\\"),
    ]
    x = np.arange(problem.n)
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 5))
    for j, (w, color, label, hatch) in enumerate(series):
        ax.bar(x + (j - 1.5) * width, w, width=width, color=color, label=label,
               edgecolor="white", linewidth=1, hatch=hatch)
    ax.axhline(0, color=INK, linewidth=1)
    ax.axhline(problem.u_max.max(), color=MUTED, linestyle="--", linewidth=1)
    ax.annotate(f"u_max = {problem.u_max.max():.2f}", xy=(problem.n - 0.5, problem.u_max.max()),
                xytext=(0, 4), textcoords="offset points", ha="right", color=MUTED, fontsize=9)
    ax.set_xticks(x, [f"{i}\nD={d:.1f}" for i, d in zip(problem.ids, problem.D)], fontsize=8)
    ax.set_ylabel("Peso $w_i$")
    ax.set_title("Pesos por instrumento", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=2)
    ax.set_ylim(min(-0.06, ax.get_ylim()[0]), problem.u_max.max() + 0.06)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    from plots import FIG_DIR

    prob = load_problem(lam=0.1)
    steps = solve_penalty(prob)
    w_pen, mu_last = steps[-1].w, steps[-1].mu

    print(f"Penalización exterior cuadrática (lam = {prob.lam}, tol interna = {INNER_TOL:g}, "
          f"máx. {INNER_MAX_ITER} iter.)")
    print_outer_table(steps)

    # Multiplicadores estimados por la penalización (último mu)
    nu, lam_lo, eta_up = penalty_multipliers(prob, w_pen, mu_last)
    at0, atu = active_sets(prob, w_pen)
    print(f"\nMultiplicadores estimados (mu = {mu_last:.0e}):")
    print(f"  nu (presupuesto) = {nu:+.8f}")
    print(f"  {'id':<5}{'w_i':>14}{'lambda_i (w>=0)':>18}{'eta_i (w<=u)':>16}  activa")
    for i, id_ in enumerate(prob.ids):
        act = "w_i = 0" if at0[i] else ("w_i = u_max" if atu[i] else "")
        print(f"  {id_:<5}{w_pen[i]:>14.3e}{lam_lo[i]:>18.6f}{eta_up[i]:>16.6f}  {act}")

    # Verificación KKT
    print("\nVerificación KKT (solución de penalización):")
    stat_pen = float(np.linalg.norm(prob.grad(w_pen) + nu - lam_lo + eta_up))
    viol = max_violation(prob, w_pen)
    min_mult = float(min(lam_lo.min(), eta_up.min()))
    print(f"  Estacionariedad con multiplicadores de penalización: ||grad L|| = {stat_pen:.2e}")
    print(f"  Factibilidad: violación máx. = {viol:.2e}  (tol {FEAS_TOL:g})")
    print(f"  Signo: min(lambda, eta) = {min_mult:.2e}  (>= 0)")
    nu_ls, lam_ls, eta_ls, resid_ls = kkt_multipliers_ls(prob, w_pen)
    inactive_mult = float(np.abs(lam_lo[~at0]).max(initial=0) + np.abs(eta_up[~atu]).max(initial=0))
    print(f"  Multiplicadores por mínimos cuadrados con el conjunto activo detectado:")
    print(f"    nu = {nu_ls:+.8f} (penalización {nu:+.8f}),  residuo estacionariedad = {resid_ls:.2e}")
    print(f"    lambda activos = {np.round(lam_ls[at0], 6)} (penalización {np.round(lam_lo[at0], 6)})")
    print(f"    eta activos    = {np.round(eta_ls[atu], 6)} (penalización {np.round(eta_up[atu], 6)})")
    print(f"  Multiplicadores de restricciones inactivas (deben ser 0): {inactive_mult:.2e}")
    assert stat_pen < 1e-6, "Estacionariedad no satisfecha"
    assert viol < FEAS_TOL, "Solución no factible"
    assert min_mult >= 0 and lam_ls.min() >= 0 and eta_ls.min() >= 0, "Multiplicador con signo incorrecto"
    assert resid_ls < 1e-6, "El conjunto activo detectado no satisface KKT"
    print("  OK: se cumplen las condiciones KKT")

    # Validación con SLSQP
    w_sq = solve_slsqp(prob)
    print("\nValidación con SLSQP (bounds + igualdad):")
    print(f"  {'':<14}{'f(w)':>14}{'gap (años)':>14}{'sum(w)':>16}{'turnover':>10}")
    for name, w in [("Penalización", w_pen), ("SLSQP", w_sq)]:
        print(f"  {name:<14}{prob.f(w):>14.10f}{prob.duration_gap(w):>+14.3e}{w.sum():>16.12f}"
              f"{turnover(prob, w):>10.4f}")
    dw = np.linalg.norm(w_pen - w_sq)
    print(f"  ||w_pen - w_SLSQP|| = {dw:.2e},  |f_pen - f_SLSQP| = {abs(prob.f(w_pen) - prob.f(w_sq)):.2e}")
    assert dw < 1e-5, "Penalización y SLSQP no coinciden"
    print("  OK: coinciden")

    # Tabla de pesos finales
    w_lagr = solve_kkt(prob).w
    print("\nPesos finales:")
    print(f"  {'id':<5}{'D_i':>7}{'w0':>9}{'w_Lagr':>9}{'':<9}{'w_pen':>9}  ")
    for i, id_ in enumerate(prob.ids):
        print(f"  {id_:<5}{prob.D[i]:>7.2f}{prob.w0[i]:>9.4f}{w_lagr[i]:>9.4f} {mark(prob, w_lagr, i):<8}"
              f"{w_pen[i]:>9.4f}  {mark(prob, w_pen, i)}")
    print(f"  Turnover:  Lagrange = {turnover(prob, w_lagr):.4f},  penalización = {turnover(prob, w_pen):.4f}")

    FIG_DIR.mkdir(exist_ok=True)
    plot_penalty_convergence(steps, FIG_DIR / "penalty_convergence.png")
    plot_weights_v2(prob, w_lagr, w_pen, FIG_DIR / "pesos_comparacion_v2.png")
    print(f"\nFiguras: {FIG_DIR / 'penalty_convergence.png'}, {FIG_DIR / 'pesos_comparacion_v2.png'}")
