"""
Método de barrera (punto interior) para el problema con presupuesto y límites de caja:

    min_w  f(w) = (D^T w - D_L)^2 + lam ||w - w0||^2
    s.a.   1^T w = 1,   0 <= w_i <= u_i  (u = u_max)

Las desigualdades se sustituyen por una barrera B(w) que tiende a +inf en el borde de la caja:
    Logarítmica:  B(w) = - sum_i [ log(w_i) + log(u_i - w_i) ]
        grad B = -1/w + 1/(u - w),        Hess B = diag( 1/w^2 + 1/(u - w)^2 )
    Inversa:      B(w) =   sum_i [ 1/w_i + 1/(u_i - w_i) ]
        grad B = -1/w^2 + 1/(u - w)^2,    Hess B = diag( 2/w^3 + 2/(u - w)^3 )

Subproblema (r > 0):   min_w  phi_r(w) = f(w) + r B(w)   s.a.  1^T w = 1.
Solver interno: Newton con restricción de igualdad; en cada iteración se resuelve el sistema KKT
    [ H   1 ] [ Δw ]   [ -g ]
    [ 1^T 0 ] [ nu ] = [  0 ],     g = grad phi_r(w),  H = Hess phi_r(w),
de modo que 1^T Δw = 0 y sum(w) = 1 se conserva en todos los iterados. El paso se limita primero
para seguir estrictamente dentro de la caja (fracción al borde 0.99) y después se aplica Armijo
sobre phi_r. Parada: decremento de Newton lambda^2 = -g^T Δw con lambda^2/2 < 1e-12, o
||Δw|| < 1e-14 (1 + ||w||), con un máximo de 100 iteraciones.

Bucle externo: r = 1, 0.1, ..., 1e-10, con warm start (el primero desde w0, estrictamente factible).

Multiplicadores (comparando grad phi_r + nu 1 = 0 con grad f + nu 1 - lambda + eta = 0):
    Logarítmica:  lambda_i ≈ r / w_i,      eta_i ≈ r / (u_i - w_i)
    Inversa:      lambda_i ≈ r / w_i^2,    eta_i ≈ r / (u_i - w_i)^2
Para la logarítmica, w_r es factible y (w_r, lambda, eta, nu) es dual factible, con brecha de
dualidad m r (m = 2n desigualdades), luego  f(w_r) - f* <= m r.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from objective import ALMProblem, load_problem
from optimizers import armijo_backtracking
from penalty import MUS, max_violation, penalty_multipliers, solve_penalty, solve_slsqp, turnover

RS = [10.0**-k for k in range(11)]  # 1, 0.1, ..., 1e-10
INNER_TOL = 1e-12                   # lambda^2 / 2 < INNER_TOL
STEP_TOL = 1e-14                    # ||Δw|| < STEP_TOL (1 + ||w||)
INNER_MAX_ITER = 100
FRAC_TO_BOUNDARY = 0.99
MU_PEN = 1e6                        # mu de la penalización usada como referencia
BARRIERS = ("log", "inv")
BARRIER_NAMES = {"log": "logarítmica", "inv": "inversa"}


@dataclass
class BarrierProblem:
    """phi_r(w) = f(w) + r B(w) con la misma interfaz (f, grad, hess) que ALMProblem."""

    base: ALMProblem
    r: float
    kind: str  # "log" o "inv"

    def slacks(self, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Holguras (w, u - w); ambas > 0 en el interior."""
        return w, self.base.u_max - w

    def barrier(self, w: np.ndarray) -> float:
        """B(w); +inf fuera del interior estricto de la caja."""
        s, t = self.slacks(w)
        if np.any(s <= 0) or np.any(t <= 0):
            return np.inf
        if self.kind == "log":
            return float(-np.sum(np.log(s)) - np.sum(np.log(t)))
        return float(np.sum(1.0 / s) + np.sum(1.0 / t))

    def f(self, w: np.ndarray) -> float:
        """phi_r(w) = f(w) + r B(w)."""
        return self.base.f(w) + self.r * self.barrier(w)

    def grad(self, w: np.ndarray) -> np.ndarray:
        """grad phi_r(w) = grad f(w) + r grad B(w)."""
        s, t = self.slacks(w)
        gB = -1.0 / s + 1.0 / t if self.kind == "log" else -1.0 / s**2 + 1.0 / t**2
        return self.base.grad(w) + self.r * gB

    def hess(self, w: np.ndarray) -> np.ndarray:
        """Hess phi_r(w) = Hess f + r Hess B(w)  (Hess B diagonal)."""
        s, t = self.slacks(w)
        hB = 1.0 / s**2 + 1.0 / t**2 if self.kind == "log" else 2.0 / s**3 + 2.0 / t**3
        return self.base.hess(w) + self.r * np.diag(hB)

    def multipliers(self, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(lambda, eta) = r (1/w, 1/(u-w)) [log] o r (1/w^2, 1/(u-w)^2) [inv]."""
        s, t = self.slacks(w)
        p = 1 if self.kind == "log" else 2
        return self.r / s**p, self.r / t**p


@dataclass
class InnerResult:
    w: np.ndarray
    nu: float           # multiplicador de 1^T w = 1 en el último sistema KKT
    n_iter: int
    stop: str           # "decremento", "paso" o "máx. iter"
    newton_dec: float   # lambda^2 / 2 en el último iterado


@dataclass
class BarrierStep:
    """Resumen de un subproblema del bucle externo."""

    r: float
    w: np.ndarray
    nu: float
    n_inner: int
    stop: str
    newton_dec: float
    min_dist: float     # min_i min(w_i, u_i - w_i)
    gap: float
    f: float
    err_sq: float       # ||w - w_SLSQP||
    cond: float         # kappa(Hess phi_r(w))
    budget: float


def check_strictly_feasible(problem: ALMProblem, w: np.ndarray, tol: float = 1e-12) -> None:
    """Lanza ValueError si no se cumple 0 < w_i < u_i para todo i y sum(w) = 1."""
    if np.any(w <= 0) or np.any(w >= problem.u_max):
        raise ValueError("El punto inicial no es estrictamente interior: se requiere 0 < w_i < u_i")
    if abs(w.sum() - 1.0) > tol:
        raise ValueError(f"El punto inicial no cumple el presupuesto: sum(w) = {w.sum():.15f}")


def max_feasible_step(bp: BarrierProblem, w: np.ndarray, dw: np.ndarray) -> float:
    """Mayor alpha en (0, 1] con w + alpha dw a fracción FRAC_TO_BOUNDARY del borde de la caja."""
    s, t = bp.slacks(w)
    ratios = np.concatenate([s[dw < 0] / -dw[dw < 0], t[dw > 0] / dw[dw > 0]])
    return min(1.0, FRAC_TO_BOUNDARY * ratios.min()) if ratios.size else 1.0


def newton_eq(bp: BarrierProblem, w_init: np.ndarray) -> InnerResult:
    """Newton con restricción de igualdad 1^T w = 1 (sistema KKT) + fracción al borde + Armijo."""
    n = bp.base.n
    w = w_init.copy()
    K = np.zeros((n + 1, n + 1))
    K[:n, n] = K[n, :n] = 1.0
    nu, dec, stop = 0.0, np.inf, "máx. iter"
    k = 0
    while k < INNER_MAX_ITER:
        g = bp.grad(w)
        K[:n, :n] = bp.hess(w)
        sol = np.linalg.solve(K, np.concatenate([-g, [0.0]]))
        dw, nu = sol[:n], float(sol[n])
        dec = 0.5 * float(-g @ dw)
        if dec < INNER_TOL:
            stop = "decremento"
            break
        if np.linalg.norm(dw) < STEP_TOL * (1.0 + np.linalg.norm(w)):
            stop = "paso"
            break
        alpha0 = max_feasible_step(bp, w, dw)
        alpha = armijo_backtracking(bp.f, bp.grad, w, bp.f(w), g, dw, alpha0=alpha0)
        w = w + alpha * dw
        k += 1
    return InnerResult(w=w, nu=nu, n_iter=k, stop=stop, newton_dec=dec)


def solve_barrier(problem: ALMProblem, kind: str, w_ref: np.ndarray,
                  rs: list[float] = RS) -> list[BarrierStep]:
    """Bucle externo de barrera con warm start desde w0."""
    w = problem.w0.copy()
    check_strictly_feasible(problem, w)
    steps = []
    for r in rs:
        bp = BarrierProblem(problem, r, kind)
        res = newton_eq(bp, w)
        w = res.w
        eig = np.linalg.eigvalsh(bp.hess(w))
        s, t = bp.slacks(w)
        steps.append(BarrierStep(
            r=r, w=w.copy(), nu=res.nu, n_inner=res.n_iter, stop=res.stop,
            newton_dec=res.newton_dec, min_dist=float(min(s.min(), t.min())),
            gap=problem.duration_gap(w), f=problem.f(w),
            err_sq=float(np.linalg.norm(w - w_ref)), cond=float(eig[-1] / eig[0]),
            budget=float(w.sum()),
        ))
    return steps


def print_barrier_table(steps: list[BarrierStep], f_ref: float, m: int | None = None) -> None:
    """Una fila por r. Si se pasa m (barrera logarítmica), añade la cota m r."""
    head = (f"  {'r':>7}{'iter':>6}{'parada':>12}{'lambda^2/2':>12}{'dist. borde':>13}"
            f"{'gap (años)':>13}{'f(w)':>14}{'||w-w_SQ||':>12}{'kappa(H)':>11}{'sum(w)':>20}")
    if m is not None:
        head += f"{'f-f_SQ':>11}{'m r':>9}  ok"
    print(head)
    for s in steps:
        line = (f"  {s.r:>7.0e}{s.n_inner:>6}{s.stop:>12}{s.newton_dec:>12.1e}{s.min_dist:>13.3e}"
                f"{s.gap:>+13.3e}{s.f:>14.10f}{s.err_sq:>12.2e}{s.cond:>11.2e}{s.budget:>20.16f}")
        if m is not None:
            d = s.f - f_ref
            line += f"{d:>11.2e}{m * s.r:>9.0e}  {'sí' if d <= m * s.r else 'NO'}"
        print(line)


def print_multipliers(problem: ALMProblem, steps: dict[str, list[BarrierStep]],
                      pen: tuple[float, np.ndarray, np.ndarray]) -> None:
    """lambda, eta (y nu) de cada barrera en el último r frente a la penalización (mu = MU_PEN)."""
    nu_pen, lam_pen, eta_pen = pen
    est = {k: BarrierProblem(problem, st[-1].r, k).multipliers(st[-1].w) for k, st in steps.items()}
    r_last = steps["log"][-1].r
    print(f"\nMultiplicadores estimados: barrera (r = {r_last:.0e}) frente a penalización (mu = {MU_PEN:.0e})")
    print(f"  nu (presupuesto):  log = {steps['log'][-1].nu:+.8f},  inv = {steps['inv'][-1].nu:+.8f},"
          f"  penalización = {nu_pen:+.8f}")
    print(f"  {'id':<5}{'lambda log':>12}{'lambda inv':>12}{'lambda pen':>12}"
          f"{'eta log':>12}{'eta inv':>12}{'eta pen':>12}")
    for i, id_ in enumerate(problem.ids):
        print(f"  {id_:<5}{est['log'][0][i]:>12.6f}{est['inv'][0][i]:>12.6f}{lam_pen[i]:>12.6f}"
              f"{est['log'][1][i]:>12.2e}{est['inv'][1][i]:>12.2e}{eta_pen[i]:>12.2e}")
    for k in BARRIERS:
        lam_b, eta_b = est[k]
        print(f"  max |lambda_{k} - lambda_pen| = {np.abs(lam_b - lam_pen).max():.2e},  "
              f"max |eta_{k} - eta_pen| = {np.abs(eta_b - eta_pen).max():.2e}")


def plot_convergence(steps: dict[str, list[BarrierStep]], path) -> None:
    """||w - w_SLSQP|| y kappa(Hess phi_r) frente a r (log, r decreciente hacia la derecha)."""
    import matplotlib.pyplot as plt
    from plots import BLUE, INK, ORANGE

    colors = {"log": BLUE, "inv": ORANGE}
    markers = {"log": "o", "inv": "s"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, attr, title, ylabel in [
        (ax1, "err_sq", "Distancia a la solución de SLSQP", r"$\|w_r - w_{\mathrm{SLSQP}}\|$"),
        (ax2, "cond", "Número de condición de la Hessiana", r"$\kappa(\nabla^2 \phi_r)$"),
    ]:
        for k, st in steps.items():
            ax.plot([s.r for s in st], [getattr(s, attr) for s in st], color=colors[k],
                    marker=markers[k], markersize=7, label=f"Barrera {BARRIER_NAMES[k]}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xlabel(r"$r$ (escala log, decreciente)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
    ax1.legend(loc="lower left")
    fig.suptitle("Método de barrera", x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_trajectories(problem: ALMProblem, pen_steps, log_steps: list[BarrierStep],
                      w_ref: np.ndarray, ids: list[str], path) -> None:
    """
    Fila superior: w_i frente al parámetro (penalización frente a mu, barrera log frente a 1/r),
    con referencias en 0, u_max y w_i de SLSQP. Fila inferior: w_i - w_i^SLSQP en escala symlog,
    para ver de qué lado se aproxima cada método (fuera: negativo en A13; dentro: positivo).
    """
    import matplotlib.pyplot as plt
    from plots import BLUE, INK, MUTED, ORANGE

    x_pen = [s.mu for s in pen_steps]
    x_bar = [1.0 / s.r for s in log_steps]
    fig, axes = plt.subplots(2, len(ids), figsize=(4.2 * len(ids), 7), sharex=True)
    for j, id_ in enumerate(ids):
        i = problem.ids.index(id_)
        y_pen = np.array([s.w[i] for s in pen_steps])
        y_bar = np.array([s.w[i] for s in log_steps])
        top, bot = axes[0, j], axes[1, j]

        for ax, yp, yb in [(top, y_pen, y_bar), (bot, y_pen - w_ref[i], y_bar - w_ref[i])]:
            ax.plot(x_pen, yp, color=BLUE, marker="o", markersize=6, label=r"Penalización (frente a $\mu$)")
            ax.plot(x_bar, yb, color=ORANGE, marker="s", markersize=6,
                    label=r"Barrera log. (frente a $1/r$)")
            ax.set_xscale("log")

        top.axhline(0, color=INK, linewidth=1)
        top.axhline(problem.u_max[i], color=MUTED, linestyle="--", linewidth=1)
        top.axhline(w_ref[i], color=MUTED, linestyle=":", linewidth=1.5)
        lo = min(y_pen.min(), y_bar.min(), 0.0)
        hi = max(y_pen.max(), y_bar.max(), problem.u_max[i])
        top.set_ylim(lo - 0.05 * (hi - lo), hi + 0.08 * (hi - lo))
        x_lab = top.get_xlim()[1]
        for y, txt, dy, va in [(problem.u_max[i], "u_max", 3, "bottom"),
                               (w_ref[i], f"SLSQP = {w_ref[i]:.4f}",
                                *((-3, "top") if w_ref[i] > problem.u_max[i] / 2 else (3, "bottom")))]:
            top.annotate(txt, xy=(x_lab, y), xytext=(-4, dy), textcoords="offset points",
                         ha="right", va=va, color=MUTED, fontsize=8)
        top.set_title(f"{id_}  (D = {problem.D[i]:.2f})", loc="left")
        top.set_ylabel("$w_i$" if j == 0 else "")

        bot.axhline(0, color=INK, linewidth=1)
        bot.set_yscale("symlog", linthresh=1e-10)
        bot.set_ylabel(r"$w_i - w_i^{\mathrm{SLSQP}}$ (symlog)" if j == 0 else "")
        bot.set_xlabel(r"$\mu$  /  $1/r$  (escala log)")
        bot.annotate("dentro / encima", xy=(0.98, 0.97), xycoords="axes fraction", va="top",
                     ha="right", fontsize=8, color=MUTED)
        bot.annotate("fuera / debajo", xy=(0.98, 0.03), xycoords="axes fraction", va="bottom",
                     ha="right", fontsize=8, color=MUTED)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, fontsize=9)
    fig.suptitle("Trayectorias: la penalización llega desde fuera, la barrera desde dentro",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    from plots import FIG_DIR

    prob = load_problem(lam=0.1)
    m = 2 * prob.n
    w_sq = solve_slsqp(prob)
    f_sq = prob.f(w_sq)

    print(f"Método de barrera (lam = {prob.lam}, n = {prob.n}, m = 2n = {m} desigualdades)")
    print(f"  w0 estrictamente factible: min(w0) = {prob.w0.min():.4f}, "
          f"min(u - w0) = {(prob.u_max - prob.w0).min():.4f}, sum(w0) = {prob.w0.sum():.15f}")
    print(f"  Interno: Newton con igualdad, fracción al borde {FRAC_TO_BOUNDARY}, Armijo; parada "
          f"lambda^2/2 < {INNER_TOL:g} o ||Δw|| < {STEP_TOL:g}(1+||w||), máx. {INNER_MAX_ITER} iter.")
    print(f"  Referencia SLSQP: f = {f_sq:.12f}")

    steps = {k: solve_barrier(prob, k, w_sq) for k in BARRIERS}
    for k, st in steps.items():
        print(f"\nBarrera {BARRIER_NAMES[k]}:")
        print_barrier_table(st, f_sq, m if k == "log" else None)
    assert all(s.f - f_sq <= m * s.r for s in steps["log"]), "Se viola la cota f(w_r) - f* <= m r"

    # Penalización: una sola ejecución con todos los mu (trayectorias) y referencia en mu = 1e6
    pen_steps = solve_penalty(prob, MUS)
    pen = next(s for s in pen_steps if s.mu == MU_PEN)
    print_multipliers(prob, steps, penalty_multipliers(prob, pen.w, MU_PEN))

    r_last = steps["log"][-1].r
    print(f"\nCota de optimalidad (barrera logarítmica, r = {r_last:.0e}):")
    d = steps["log"][-1].f - f_sq
    print(f"  f(w_r) - f(w_SLSQP) = {d:.3e}  <=  m r = {m * r_last:.1e}:  {'sí' if d <= m * r_last else 'NO'}")

    # Tabla comparativa final
    n_pen = sum(s.n_inner for s in pen_steps if s.mu <= MU_PEN)
    rows = [
        (f"Penalización (mu={MU_PEN:.0e})", pen.w, str(n_pen)),
        ("Barrera logarítmica", steps["log"][-1].w, str(sum(s.n_inner for s in steps["log"]))),
        ("Barrera inversa", steps["inv"][-1].w, str(sum(s.n_inner for s in steps["inv"]))),
        ("SLSQP", w_sq, "—"),
    ]
    print("\nComparación final:")
    print(f"  {'Método':<26}{'f(w)':>16}{'gap (años)':>13}{'sum(w)':>20}{'viol. máx':>11}"
          f"{'turnover':>10}{'iter. Newton':>14}")
    for name, w, it in rows:
        print(f"  {name:<26}{prob.f(w):>16.12f}{prob.duration_gap(w):>+13.4e}{w.sum():>20.16f}"
              f"{max_violation(prob, w):>11.2e}{turnover(prob, w):>10.6f}{it:>14}")

    FIG_DIR.mkdir(exist_ok=True)
    plot_convergence(steps, FIG_DIR / "interior_point_convergence.png")
    plot_trajectories(prob, pen_steps, steps["log"], w_sq, ["A13", "A15", "A01"],
                      FIG_DIR / "trayectorias.png")
    print(f"\nFiguras: {FIG_DIR / 'interior_point_convergence.png'}, {FIG_DIR / 'trayectorias.png'}")
