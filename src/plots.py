"""
Figuras del proyecto de duration matching (se guardan en figures/):

- convergencia.png:       log10 ||grad f(w_k)|| frente a k (gradiente 1/L, gradiente Armijo,
                          Newton) desde la cartera uniforme.
- lambda_sweep.png:       gap residual y número de condición kappa(H) = 1 + ||D||^2 / lam del
                          problema sin restricciones frente a lam (eje x logarítmico).
- pesos_comparacion.png:  pesos por instrumento para w0, el óptimo sin restricciones y el
                          óptimo de Lagrange (1^T w = 1), con referencias en 0 y u_max.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lagrange import solve_kkt
from objective import ALMProblem, load_problem
from optimizers import closed_form_solution, gradient_descent, newton

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

# Paleta categórica validada (orden fijo) y tinta neutra para texto/ejes.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#2b2b29", "#6b6a63", "#e4e3dc"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.frameon": False,
    "lines.linewidth": 2,
})


def plot_convergence(problem: ALMProblem, path: Path) -> None:
    """log10 ||grad f(w_k)|| frente a la iteración k, desde la cartera uniforme 1/n."""
    w_init = np.full(problem.n, 1.0 / problem.n)
    runs = [
        (gradient_descent(problem, w_init, step="fixed"), BLUE, "Gradiente (paso 1/L)"),
        (gradient_descent(problem, w_init, step="armijo"), ORANGE, "Gradiente (Armijo)"),
        (newton(problem, w_init), AQUA, "Newton"),
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for res, color, label in runs:
        k = np.arange(len(res.grad_norm_hist))
        y = np.log10(res.grad_norm_hist)
        marker = "o" if len(k) < 10 else None
        ax.plot(k, y, color=color, marker=marker, markersize=8,
                label=f"{label}: {res.n_iter:,} iter".replace(",", " "))
        print(f"  {label:<22} iter = {res.n_iter:>7}   ||grad|| final = {res.grad_norm:.2e}")
    ax.axhline(np.log10(1e-8), color=MUTED, linestyle="--", linewidth=1)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlim(left=0)
    ax.annotate("tol = 1e-8", xy=(2, -8), xytext=(0, 4),
                textcoords="offset points", ha="left", color=MUTED, fontsize=9)
    ax.set_xlabel("Iteración k (escala symlog)")
    ax.set_ylabel(r"$\log_{10}\,\|\nabla f(w_k)\|$")
    ax.set_title("Convergencia desde la cartera uniforme", loc="left")
    ax.legend(loc="lower left", bbox_to_anchor=(0.25, 0.03))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_lambda_sweep(lams: list[float], path: Path) -> None:
    """Gap residual |D^T w*(lam) - D_L| y kappa(H(lam)) del problema sin restricciones."""
    gaps, conds = [], []
    print(f"  {'lam':>8}{'gap residual':>16}{'kappa(H)':>14}{'sum(w*)':>10}")
    for lam in lams:
        prob = load_problem(lam=lam)
        w_star = closed_form_solution(prob)
        eig = np.linalg.eigvalsh(prob.hess(w_star))
        gaps.append(abs(prob.duration_gap(w_star)))
        conds.append(eig[-1] / eig[0])
        print(f"  {lam:>8g}{gaps[-1]:>16.6e}{conds[-1]:>14.1f}{w_star.sum():>10.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, y, title, ylabel in [
        (ax1, gaps, "Gap residual", r"$|D^\top w^* - D_L|$ (años)"),
        (ax2, conds, "Número de condición de la Hessiana", r"$\kappa(H)$"),
    ]:
        ax.plot(lams, y, color=BLUE, marker="o", markersize=8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\lambda$ (escala log)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        for x, v in zip(lams, y):
            ax.annotate(f"{v:.2g}", (x, v), xytext=(6, 4), textcoords="offset points",
                        fontsize=8, color=INK)
    fig.suptitle(r"Efecto de $\lambda$ (problema sin restricciones, solución exacta)",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_weights(problem: ALMProblem, path: Path) -> None:
    """Barras agrupadas de pesos: w0, óptimo sin restricciones y óptimo de Lagrange."""
    series = [
        (problem.w0, BLUE, "Cartera actual $w_0$", None),
        (closed_form_solution(problem), ORANGE, "Óptimo sin restricciones", None),
        (solve_kkt(problem).w, AQUA, r"Óptimo Lagrange ($\mathbf{1}^\top w = 1$)", "////"),
    ]
    x = np.arange(problem.n)
    width = 0.27

    fig, ax = plt.subplots(figsize=(11, 4.8))
    for j, (w, color, label, hatch) in enumerate(series):
        ax.bar(x + (j - 1) * width, w, width=width, color=color, label=label,
               edgecolor="white", linewidth=1, hatch=hatch)
    ax.axhline(0, color=INK, linewidth=1)
    ax.axhline(problem.u_max.max(), color=MUTED, linestyle="--", linewidth=1)
    ax.annotate(f"u_max = {problem.u_max.max():.2f}", xy=(problem.n - 0.5, problem.u_max.max()),
                xytext=(0, 4), textcoords="offset points", ha="right", color=MUTED, fontsize=9)
    ax.set_xticks(x, [f"{i}\nD={d:.1f}" for i, d in zip(problem.ids, problem.D)], fontsize=8)
    ax.set_ylabel("Peso $w_i$")
    ax.set_title("Pesos por instrumento", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=3)
    ax.set_ylim(min(-0.06, ax.get_ylim()[0]), problem.u_max.max() + 0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    FIG_DIR.mkdir(exist_ok=True)
    prob = load_problem(lam=0.1)

    print("convergencia.png")
    plot_convergence(prob, FIG_DIR / "convergencia.png")
    print("lambda_sweep.png")
    plot_lambda_sweep([0.001, 0.01, 0.1, 1.0], FIG_DIR / "lambda_sweep.png")
    print("pesos_comparacion.png")
    plot_weights(prob, FIG_DIR / "pesos_comparacion.png")
    print(f"\nFiguras guardadas en {FIG_DIR}")
