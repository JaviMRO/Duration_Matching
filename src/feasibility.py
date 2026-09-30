"""
Chequeo de factibilidad del objetivo de duración.

Se calcula el rango de duraciones de cartera alcanzables resolviendo dos programas lineales:

    D_min = min_w  D^T w      s.a.  1^T w = 1,  0 <= w_i <= u_max_i
    D_max = max_w  D^T w      s.a.  (mismas restricciones)

El objetivo D_L es alcanzable con gap nulo si y solo si D_min <= D_L <= D_max
(el conjunto factible es un politopo convexo y D^T w es continua sobre él).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from objective import ALMProblem, load_problem

SAT_TOL = 1e-9  # tolerancia para considerar que un peso está en su límite


@dataclass
class FeasibilityResult:
    """Extremos alcanzables de la duración y carteras que los alcanzan."""

    D_min: float
    D_max: float
    w_min: np.ndarray
    w_max: np.ndarray
    D_L: float

    @property
    def D_L_feasible(self) -> bool:
        """True si D_min <= D_L <= D_max."""
        return self.D_min <= self.D_L <= self.D_max


def _solve_extreme(problem: ALMProblem, sign: float) -> np.ndarray:
    """Resuelve min_w sign * D^T w  s.a. 1^T w = 1, 0 <= w <= u_max (sign=+1 mín., -1 máx.)."""
    res = linprog(
        c=sign * problem.D,
        A_eq=np.ones((1, problem.n)),
        b_eq=[1.0],
        bounds=list(zip(np.zeros(problem.n), problem.u_max)),
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"linprog falló: {res.message}")
    return res.x


def check_feasibility(problem: ALMProblem) -> FeasibilityResult:
    """Calcula [D_min, D_max] con dos PL y devuelve las carteras extremas."""
    w_min = _solve_extreme(problem, +1.0)
    w_max = _solve_extreme(problem, -1.0)
    return FeasibilityResult(
        D_min=float(problem.D @ w_min),
        D_max=float(problem.D @ w_max),
        w_min=w_min,
        w_max=w_max,
        D_L=problem.D_L,
    )


def saturated(problem: ALMProblem, w: np.ndarray) -> tuple[list[int], list[int], list[int]]:
    """Índices de instrumentos en el límite superior, en el inferior (0) y estrictamente dentro."""
    upper = [i for i in range(problem.n) if w[i] >= problem.u_max[i] - SAT_TOL]
    lower = [i for i in range(problem.n) if w[i] <= SAT_TOL]
    inner = [i for i in range(problem.n) if i not in upper and i not in lower]
    return upper, lower, inner


def _print_extreme(problem: ALMProblem, name: str, w: np.ndarray, D_val: float) -> None:
    """Muestra la composición de una cartera extrema."""
    upper, lower, inner = saturated(problem, w)
    fmt = lambda idx: ", ".join(f"{problem.ids[i]} (D={problem.D[i]:.2f})" for i in idx) or "-"
    print(f"\n{name}: D^T w = {D_val:.4f} años")
    print(f"  En u_max ({len(upper)}): {fmt(upper)}")
    print(f"  Interior ({len(inner)}): "
          + (", ".join(f"{problem.ids[i]} (D={problem.D[i]:.2f}, w={w[i]:.4f})" for i in inner)
             or "-"))
    print(f"  En 0     ({len(lower)}): {fmt(lower)}")


if __name__ == "__main__":
    prob = load_problem()
    res = check_feasibility(prob)

    print("Rango de duraciones alcanzables con 1^T w = 1, 0 <= w <= u_max:")
    print(f"  D_min = {res.D_min:.4f} años")
    print(f"  D_max = {res.D_max:.4f} años")
    print(f"  D_L   = {res.D_L:.4f} años  ->  "
          + ("DENTRO del intervalo: el gap se puede cerrar por completo"
             if res.D_L_feasible else "FUERA del intervalo: no es posible gap nulo"))
    print(f"  Holgura: D_L - D_min = {res.D_L - res.D_min:.4f},  D_max - D_L = {res.D_max - res.D_L:.4f}")

    _print_extreme(prob, "Cartera de duración mínima", res.w_min, res.D_min)
    _print_extreme(prob, "Cartera de duración máxima", res.w_max, res.D_max)
