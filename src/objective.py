"""
Función objetivo del problema de duration matching (ALM bancario).

Problema:
    min_w  f(w) = (D^T w - D_L)^2 + lam * ||w - w0||^2

donde
    w   in R^n : pesos de la cartera de activos,
    D   in R^n : duraciones modificadas de los instrumentos,
    D_L        : duración objetivo de los pasivos,
    w0         : pesos de la cartera actual,
    lam > 0    : peso del término de proximidad a la cartera actual.

Sin el término de proximidad la Hessiana 2 D D^T tiene rango 1 (singular) y Newton no está bien definido; con lam > 0 pasa a ser definida positiva.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class ALMProblem:
    """Datos del problema de duration matching y sus derivadas analíticas."""

    D: np.ndarray        # duraciones modificadas D_i, shape (n,)
    w0: np.ndarray       # pesos actuales, shape (n,), suman 1
    u_max: np.ndarray    # límite superior por instrumento, shape (n,)
    D_L: float           # duración objetivo de los pasivos
    lam: float           # peso del término de proximidad
    ids: list[str]       # identificadores de los instrumentos

    @property
    def n(self) -> int:
        """Número de instrumentos."""
        return self.D.size

    def duration_gap(self, w: np.ndarray) -> float:
        """Gap de duración: g(w) = D^T w - D_L (en años)."""
        return float(self.D @ w - self.D_L)

    def f(self, w: np.ndarray) -> float:
        """Función objetivo: f(w) = (D^T w - D_L)^2 + lam * ||w - w0||^2."""
        gap = self.duration_gap(w)
        diff = w - self.w0
        return gap**2 + self.lam * float(diff @ diff)

    def grad(self, w: np.ndarray) -> np.ndarray:
        """Gradiente: grad f(w) = 2 (D^T w - D_L) D + 2 lam (w - w0)."""
        return 2.0 * self.duration_gap(w) * self.D + 2.0 * self.lam * (w - self.w0)

    def hess(self, w: np.ndarray) -> np.ndarray:
        """Hessiana: Hess f(w) = 2 D D^T + 2 lam I (constante, no depende de w)."""
        return 2.0 * np.outer(self.D, self.D) + 2.0 * self.lam * np.eye(self.n)


def load_problem(lam: float = 0.1) -> ALMProblem:
    """Lee data/assets.csv y data/summary.json y construye el ALMProblem."""
    assets = pd.read_csv(DATA_DIR / "assets.csv")
    with open(DATA_DIR / "summary.json", encoding="utf-8") as fh:
        summary = json.load(fh)

    return ALMProblem(
        D=assets["duracion_modificada"].to_numpy(dtype=float),
        w0=assets["w0_actual"].to_numpy(dtype=float),
        u_max=assets["u_max"].to_numpy(dtype=float),
        D_L=float(summary["duracion_objetivo_pasivos_D_L"]),
        lam=float(lam),
        ids=assets["instrument_id"].astype(str).tolist(),
    )


def _random_points(problem: ALMProblem, n_trials: int, seed: int = 0) -> np.ndarray:
    """Puntos de prueba aleatorios en el simplex (Dirichlet), con semilla fija."""
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(problem.n), size=n_trials)


def _rel_error(approx: np.ndarray, exact: np.ndarray) -> float:
    """Error relativo ||approx - exact|| / max(||exact||, 1e-12)."""
    return float(np.linalg.norm(approx - exact) / max(np.linalg.norm(exact), 1e-12))


def check_gradient(problem: ALMProblem, n_trials: int = 5, h: float = 1e-6) -> float:
    """
    Valida grad(w) con diferencias centrales de f:
        [grad f(w)]_i ≈ (f(w + h e_i) - f(w - h e_i)) / (2h).
    Devuelve el error relativo máximo sobre n_trials puntos aleatorios.
    """
    errors = []
    I = np.eye(problem.n)
    for w in _random_points(problem, n_trials):
        fd = np.array([(problem.f(w + h * e) - problem.f(w - h * e)) / (2 * h) for e in I])
        errors.append(_rel_error(fd, problem.grad(w)))
    return max(errors)


def check_hessian(problem: ALMProblem, n_trials: int = 5, h: float = 1e-6) -> float:
    """
    Valida hess(w) con diferencias centrales del gradiente (columna a columna):
        [Hess f(w)]_{:, j} ≈ (grad f(w + h e_j) - grad f(w - h e_j)) / (2h).
    Devuelve el error relativo máximo (norma de Frobenius) sobre n_trials puntos.
    """
    errors = []
    I = np.eye(problem.n)
    for w in _random_points(problem, n_trials, seed=1):
        fd = np.column_stack(
            [(problem.grad(w + h * e) - problem.grad(w - h * e)) / (2 * h) for e in I]
        )
        errors.append(_rel_error(fd, problem.hess(w)))
    return max(errors)


if __name__ == "__main__":
    prob = load_problem(lam=0.1)
    w0 = prob.w0

    print(f"Problema: n = {prob.n} instrumentos, D_L = {prob.D_L}, lam = {prob.lam}")
    print(f"  sum(w0)          = {w0.sum():.6f}")
    print(f"  f(w0)            = {prob.f(w0):.6f}")
    print(f"  gap inicial      = {prob.duration_gap(w0):+.4f} años")
    print(f"  ||grad f(w0)||   = {np.linalg.norm(prob.grad(w0)):.6f}")

    err_g = check_gradient(prob)
    err_h = check_hessian(prob)
    print("\nValidación por diferencias finitas (centrales, h = 1e-6):")
    print(f"  error relativo gradiente = {err_g:.3e}")
    print(f"  error relativo Hessiana  = {err_h:.3e}")
    assert err_g < 1e-5, f"Gradiente incorrecto: error {err_g:.3e}"
    assert err_h < 1e-5, f"Hessiana incorrecta: error {err_h:.3e}"
    print("  OK: ambos errores < 1e-5")

    eig = np.linalg.eigvalsh(prob.hess(w0))
    lmin, lmax = eig[0], eig[-1]
    print("\nEspectro de la Hessiana:")
    print(f"  lambda_min = {lmin:.6f}  (teórico 2*lam = {2 * prob.lam:.6f})")
    print(f"  lambda_max = {lmax:.6f}  (teórico 2*||D||^2 + 2*lam = "
          f"{2 * prob.D @ prob.D + 2 * prob.lam:.6f})")
    print(f"  número de condición = {lmax / lmin:.2f}")
    assert lmin > 0, "La Hessiana no es definida positiva"
    print("  OK: Hessiana definida positiva")
