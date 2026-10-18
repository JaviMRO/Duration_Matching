"""
Restricción de rebalanceo (turnover) sobre el problema con presupuesto y límites de caja:

    min_w  f(w) = (D^T w - D_L)^2 + lam ||w - w0||^2
    s.a.   1^T w = 1,   0 <= w_i <= u_i,   sum_i |w_i - w0_i| <= T_max

Transformación positiva-negativa: w = w0 + p - q con p, q >= 0 (compras y ventas). Como 1^T w0 = 1,
en x = (p, q) in R^{2n} el problema queda

    min_x  F(x) = f(w0 + p - q)
    s.a.   A x = 0,   A = [1^T, -1^T]                       (presupuesto)
           0 <= p_i <= c_i = u_i - w0_i,   0 <= q_i <= w0_i  (implican 0 <= w_i <= u_i)
           1^T p + 1^T q <= T_max                           (turnover)

Es equivalente al original: todo w factible se representa con p = max(w - w0, 0),
q = max(w0 - w, 0), que cumple las cotas y sum(p + q) = sum|w - w0|; y para todo x factible,
sum|w - w0| <= sum(p + q) <= T_max. La igualdad solo se da si p_i q_i = 0: cuando la restricción de
turnover no está activa, (p, q) no es único (F solo depende de p - q) y sum(p + q) sobreestima el
turnover, que por eso se calcula siempre a partir de w.

    grad F = [g; -g],   Hess F = [[H, -H], [-H, H]]   (g = grad f(w), H = Hess f; singular en (e_i, e_i))

Barrera logarítmica sobre las 4n + 1 desigualdades (s = T_max - 1^T p - 1^T q):
    B(x) = - sum log p - sum log(c - p) - sum log q - sum log(w0 - q) - log s
    grad B = [ -1/p + 1/(c - p);  -1/q + 1/(w0 - q) ] + (1/s) 1
    Hess B = diag( 1/p^2 + 1/(c - p)^2,  1/q^2 + 1/(w0 - q)^2 ) + (1/s^2) 1 1^T
La curvatura de r B hace definida positiva la Hessiana de phi_r = F + r B también en la dirección
(e_i, e_i), en la que F es plana.

Solver interno: Newton con restricción de igualdad A x = 0 (sistema KKT), fracción al borde 0.99,
Armijo y parada por decremento de Newton (mismos parámetros que interior_point.py). Bucle externo
r = 1, 0.1, ..., 1e-10 con warm start, desde p = q = delta,
    delta = 0.5 min( min(u - w0), min(w0), T_max / (2n) )   (estrictamente factible: sum(p+q) <= T_max/2).

Multiplicadores (estacionariedad de la lagrangiana):
    [g; -g] + nu [1; -1] - [lambda_p; lambda_q] + [eta_p; eta_q] + tau 1 = 0
    lambda_p ≈ r/p,  eta_p ≈ r/(c - p),  lambda_q ≈ r/q,  eta_q ≈ r/(w0 - q),  tau ≈ r/s
tau es el precio sombra del turnover: tau = -df*/dT_max. Brecha de dualidad: f(w_r) - f* <= (4n+1) r.

La estimación tau ≈ r/s es muy sensible a lo centrado que esté el iterado: d(r/s)/ds = -tau^2 / r.
Para r <= 1e-8 la curvatura de la barrera en la dirección de la holgura (r/s^2 = tau^2/r ~ 1e10) hace
que el decremento de Newton caiga por debajo de 1e-12 sin que s esté centrado (x es correcto, pero s
no está sobre el camino central) y r/s queda desplazado ~1e-3 relativo. Por eso se da también la
estimación corregida con el último paso de Newton (el mismo sistema KKT del que sale nu):
    tau ≈ r / (s + Δs),   Δs = -1^T Δx.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog, minimize

from interior_point import FRAC_TO_BOUNDARY, INNER_MAX_ITER, INNER_TOL, RS, STEP_TOL
from objective import ALMProblem, load_problem
from optimizers import armijo_backtracking
from penalty import ACTIVE_TOL, kkt_multipliers_ls, solve_penalty, turnover

T_GRID = [round(0.1 * k, 1) for k in range(2, 11)]          # 0.2, 0.3, ..., 1.0
T_FINE = [round(0.01 * k, 2) for k in range(1, 111)]        # 0.01, ..., 1.10 (figura y orden de venta)
FD_H = 1e-3            # paso de las diferencias finitas centrales de f*(T_max)
CMP_TOL = 1e-5         # tolerancia de comparación con SLSQP y con penalty.py
SIGN_TOL = 1e-8        # multiplicador >= -SIGN_TOL se acepta como no negativo
KKT_TOL = 1e-6         # residuo de estacionariedad máximo en la verificación KKT
TO_PEN_REF = 0.9552    # turnover de la solución de penalty.py (sin límite de turnover)
SELL_ORDER = ["A14", "A13", "A11", "A15", "A04"]  # duración decreciente


def to_w(problem: ALMProblem, x: np.ndarray) -> np.ndarray:
    """w = w0 + p - q, con x = (p, q)."""
    n = problem.n
    return problem.w0 + x[:n] - x[n:]


def pq_bounds(problem: ALMProblem) -> list[tuple[float, float]]:
    """Cotas 0 <= p <= u - w0, 0 <= q <= w0."""
    zeros = np.zeros(problem.n)
    return list(zip(zeros, problem.u_max - problem.w0)) + list(zip(zeros, problem.w0))


# ------------------------------------------------------------------------------------------------
# Partes 1 y 2: programas lineales
# ------------------------------------------------------------------------------------------------

def min_turnover_lp(problem: ALMProblem) -> np.ndarray:
    """min 1^T(p + q)  s.a.  D^T w = D_L,  A x = 0,  cotas de p y q.  Devuelve x = (p, q)."""
    n = problem.n
    ones = np.ones(n)
    res = linprog(
        c=np.ones(2 * n),
        A_eq=np.vstack([np.concatenate([problem.D, -problem.D]), np.concatenate([ones, -ones])]),
        b_eq=[problem.D_L - problem.D @ problem.w0, 0.0],
        bounds=pq_bounds(problem),
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"linprog falló: {res.message}")
    return res.x


def frontier_lp(problem: ALMProblem, T_max: float) -> np.ndarray:
    """
    min t  s.a.  -t <= D^T w - D_L <= t,  A x = 0,  1^T(p + q) <= T_max,  cotas.
    Variables (p, q, t); devuelve x = (p, q).
    """
    n = problem.n
    ones = np.ones(n)
    dD = np.concatenate([problem.D, -problem.D])
    b = problem.D_L - problem.D @ problem.w0          # D^T w - D_L = dD^T x - b
    res = linprog(
        c=np.concatenate([np.zeros(2 * n), [1.0]]),
        A_ub=np.vstack([np.concatenate([dD, [-1.0]]), np.concatenate([-dD, [-1.0]]),
                        np.concatenate([np.ones(2 * n), [0.0]])]),
        b_ub=[b, -b, T_max],
        A_eq=np.concatenate([ones, -ones, [0.0]])[None, :],
        b_eq=[0.0],
        bounds=pq_bounds(problem) + [(0.0, None)],
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"linprog falló (T_max = {T_max}): {res.message}")
    return res.x[:2 * n]


# ------------------------------------------------------------------------------------------------
# Parte 3: barrera logarítmica en (p, q)
# ------------------------------------------------------------------------------------------------

@dataclass
class TurnoverBarrier:
    """phi_r(x) = F(x) + r B(x) en x = (p, q), con la misma interfaz (f, grad, hess) que ALMProblem."""

    base: ALMProblem
    T_max: float
    r: float

    def slacks(self, x: np.ndarray) -> tuple[np.ndarray, ...]:
        """(p, c - p, q, w0 - q, [T_max - 1^T(p + q)]); todas > 0 en el interior."""
        n, w0 = self.base.n, self.base.w0
        p, q = x[:n], x[n:]
        return p, self.base.u_max - w0 - p, q, w0 - q, np.array([self.T_max - x.sum()])

    def barrier(self, x: np.ndarray) -> float:
        """B(x); +inf fuera del interior estricto."""
        sl = self.slacks(x)
        if any(np.any(s <= 0) for s in sl):
            return np.inf
        return float(-sum(np.sum(np.log(s)) for s in sl))

    def f(self, x: np.ndarray) -> float:
        """phi_r(x) = f(w0 + p - q) + r B(x)."""
        return self.base.f(to_w(self.base, x)) + self.r * self.barrier(x)

    def grad(self, x: np.ndarray) -> np.ndarray:
        """grad phi_r(x) = [g; -g] + r grad B(x)."""
        p, cp, q, cq, s = self.slacks(x)
        g = self.base.grad(to_w(self.base, x))
        gB = np.concatenate([-1.0 / p + 1.0 / cp, -1.0 / q + 1.0 / cq]) + 1.0 / s[0]
        return np.concatenate([g, -g]) + self.r * gB

    def hess(self, x: np.ndarray) -> np.ndarray:
        """Hess phi_r(x) = [[H, -H], [-H, H]] + r (diag(...) + 1 1^T / s^2)."""
        p, cp, q, cq, s = self.slacks(x)
        H = self.base.hess(to_w(self.base, x))
        hB = np.diag(np.concatenate([1.0 / p**2 + 1.0 / cp**2, 1.0 / q**2 + 1.0 / cq**2])) + 1.0 / s[0]**2
        return np.block([[H, -H], [-H, H]]) + self.r * hB

    def multipliers(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """(lambda_p, eta_p, lambda_q, eta_q, tau) = r / holguras."""
        lp, ep, lq, eq, s = (self.r / s for s in self.slacks(x))
        return lp, ep, lq, eq, float(s[0])


@dataclass
class InnerResult:
    x: np.ndarray
    nu: float           # multiplicador de A x = 0 en el último sistema KKT
    n_iter: int
    stop: str           # "decremento", "paso" o "máx. iter"
    newton_dec: float   # lambda^2 / 2 en el último iterado
    dx: np.ndarray      # último paso de Newton calculado (no aplicado si paró)


@dataclass
class TurnoverSolution:
    """Resultado del bucle externo de barrera para un T_max."""

    T_max: float
    x: np.ndarray
    w: np.ndarray
    nu: float
    r: float            # último r
    n_inner: int        # iteraciones de Newton acumuladas
    stop: str           # parada del último subproblema
    newton_dec: float
    tau_path: list[tuple[float, float, float]]  # (r, r/s, r/(s + Δs)) en cada subproblema


def initial_point(problem: ALMProblem, T_max: float) -> np.ndarray:
    """p = q = delta = 0.5 min( min(u - w0), min(w0), T_max / (2n) )."""
    delta = 0.5 * min((problem.u_max - problem.w0).min(), problem.w0.min(), T_max / (2 * problem.n))
    return np.full(2 * problem.n, delta)


def check_strictly_feasible(bp: TurnoverBarrier, x: np.ndarray, tol: float = 1e-12) -> None:
    """Lanza ValueError si alguna holgura no es > 0 o si A x != 0."""
    if any(np.any(s <= 0) for s in bp.slacks(x)):
        raise ValueError("El punto inicial no es estrictamente interior")
    n = bp.base.n
    if abs(x[:n].sum() - x[n:].sum()) > tol:
        raise ValueError(f"El punto inicial no cumple A x = 0: {x[:n].sum() - x[n:].sum():.3e}")


def max_feasible_step(bp: TurnoverBarrier, x: np.ndarray, dx: np.ndarray) -> float:
    """Mayor alpha en (0, 1] con x + alpha dx a fracción FRAC_TO_BOUNDARY del borde (todas las holguras)."""
    n = bp.base.n
    dp, dq = dx[:n], dx[n:]
    d_slacks = (dp, -dp, dq, -dq, np.array([-dx.sum()]))
    ratios = np.concatenate([s[d < 0] / -d[d < 0] for s, d in zip(bp.slacks(x), d_slacks)])
    return min(1.0, FRAC_TO_BOUNDARY * ratios.min()) if ratios.size else 1.0


def newton_eq_pq(bp: TurnoverBarrier, x_init: np.ndarray) -> InnerResult:
    """Newton con restricción de igualdad A x = 0 (sistema KKT) + fracción al borde + Armijo."""
    n = bp.base.n
    m = 2 * n
    A = np.concatenate([np.ones(n), -np.ones(n)])
    x = x_init.copy()
    K = np.zeros((m + 1, m + 1))
    K[:m, m] = K[m, :m] = A
    nu, dec, stop = 0.0, np.inf, "máx. iter"
    k = 0
    while k < INNER_MAX_ITER:
        g = bp.grad(x)
        K[:m, :m] = bp.hess(x)
        sol = np.linalg.solve(K, np.concatenate([-g, [0.0]]))
        dx, nu = sol[:m], float(sol[m])
        dec = 0.5 * float(-g @ dx)
        if dec < INNER_TOL:
            stop = "decremento"
            break
        if np.linalg.norm(dx) < STEP_TOL * (1.0 + np.linalg.norm(x)):
            stop = "paso"
            break
        alpha0 = max_feasible_step(bp, x, dx)
        alpha = armijo_backtracking(bp.f, bp.grad, x, bp.f(x), g, dx, alpha0=alpha0)
        x = x + alpha * dx
        k += 1
    return InnerResult(x=x, nu=nu, n_iter=k, stop=stop, newton_dec=dec, dx=dx)


def solve_turnover(problem: ALMProblem, T_max: float, rs: list[float] = RS) -> TurnoverSolution:
    """Bucle externo de barrera con warm start desde p = q = delta."""
    x = initial_point(problem, T_max)
    check_strictly_feasible(TurnoverBarrier(problem, T_max, rs[0]), x)
    n_inner, tau_path = 0, []
    for r in rs:
        bp = TurnoverBarrier(problem, T_max, r)
        res = newton_eq_pq(bp, x)
        x = res.x
        n_inner += res.n_iter
        s = T_max - x.sum()
        tau_path.append((r, r / s, r / (s - res.dx.sum())))
    return TurnoverSolution(T_max=T_max, x=x, w=to_w(problem, x), nu=res.nu, r=rs[-1],
                            n_inner=n_inner, stop=res.stop, newton_dec=res.newton_dec,
                            tau_path=tau_path)


def solve_slsqp_pq(problem: ALMProblem, T_max: float, x_init: np.ndarray):
    """Referencia: SLSQP en (p, q) con A x = 0, 1^T x <= T_max y cotas, desde x_init."""
    n = problem.n
    A = np.concatenate([np.ones(n), -np.ones(n)])

    def jac(x: np.ndarray) -> np.ndarray:
        g = problem.grad(to_w(problem, x))
        return np.concatenate([g, -g])

    return minimize(
        lambda x: problem.f(to_w(problem, x)), x_init, jac=jac, method="SLSQP",
        bounds=pq_bounds(problem),
        constraints=[{"type": "eq", "fun": lambda x: A @ x, "jac": lambda x: A},
                     {"type": "ineq", "fun": lambda x: T_max - x.sum(), "jac": lambda x: -np.ones(2 * n)}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )


# ------------------------------------------------------------------------------------------------
# Partes 4 y 5: precio sombra, conjunto activo y verificación KKT
# ------------------------------------------------------------------------------------------------

@dataclass
class KKTCheck:
    space: str          # "(p,q)" o "w"
    nu: float
    tau: float          # 0 si se verifica en el espacio de w
    min_mult: float     # mínimo de los multiplicadores de desigualdad estimados
    resid: float        # residuo de estacionariedad
    n_active: int

    @property
    def ok(self) -> bool:
        return self.resid < KKT_TOL and self.min_mult >= -SIGN_TOL


@dataclass
class TurnoverCase:
    """Todo lo que se guarda para un T_max."""

    sol: TurnoverSolution
    f: float
    gap: float
    turnover: float     # real: sum |w - w0|
    sum_pq: float       # 1^T(p + q) (>= turnover real)
    active: bool        # restricción de turnover activa
    tau_bar: float      # r / s en el último r
    tau_newton: float   # r / (s + Δs) en el último r
    tau_fd: float       # (f*(T - h) - f*(T + h)) / (2h)
    f_sq: float
    dw_sq: float        # ||w - w_SLSQP||
    sq_success: bool
    sq_message: str
    labels: str         # restricciones activas
    kkt: KKTCheck


def pq_active_sets(problem: ALMProblem, x: np.ndarray, tol: float = ACTIVE_TOL):
    """Máscaras (p = 0, p = u - w0, q = 0, q = w0)."""
    n = problem.n
    p, q = x[:n], x[n:]
    return p <= tol, problem.u_max - problem.w0 - p <= tol, q <= tol, problem.w0 - q <= tol


def kkt_check_pq(problem: ALMProblem, x: np.ndarray) -> KKTCheck:
    """
    Con turnover activo: fija el conjunto activo detectado en x y resuelve por mínimos cuadrados
        [g; -g] + nu [1; -1] + tau 1 - sum_{p_i=0} lambda_p,i e_i + sum_{p_i=c_i} eta_p,i e_i
                                     - sum_{q_i=0} lambda_q,i e_{n+i} + sum_{q_i=w0_i} eta_q,i e_{n+i} = 0.
    """
    n = problem.n
    g = problem.grad(to_w(problem, x))
    G = np.concatenate([g, -g])
    I = np.eye(2 * n)
    p0, pu, q0, qu = pq_active_sets(problem, x)
    cols = [np.concatenate([np.ones(n), -np.ones(n)]), np.ones(2 * n)]
    cols += [-I[i] for i in np.where(p0)[0]] + [I[i] for i in np.where(pu)[0]]
    cols += [-I[n + i] for i in np.where(q0)[0]] + [I[n + i] for i in np.where(qu)[0]]
    M = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(M, -G, rcond=None)
    return KKTCheck(space="(p,q)", nu=float(coef[0]), tau=float(coef[1]), min_mult=float(coef[1:].min()),
                    resid=float(np.linalg.norm(G + M @ coef)), n_active=M.shape[1] - 1)


def kkt_check_w(problem: ALMProblem, w: np.ndarray) -> KKTCheck:
    """Turnover inactivo: verificación en el espacio de w (kkt_multipliers_ls de penalty.py)."""
    nu, lam_, eta, resid = kkt_multipliers_ls(problem, w)
    at0 = np.abs(w) <= ACTIVE_TOL
    atu = np.abs(w - problem.u_max) <= ACTIVE_TOL
    mults = np.concatenate([lam_[at0], eta[atu]])
    return KKTCheck(space="w", nu=nu, tau=0.0, min_mult=float(mults.min(initial=0.0)), resid=resid,
                    n_active=int(at0.sum() + atu.sum()))


def active_labels(problem: ALMProblem, w: np.ndarray, active: bool) -> str:
    """Restricciones activas descritas en w: w_i = 0, w_i = u_i, w_i = w0_i (p_i = q_i = 0), turnover."""
    ids = problem.ids
    groups = [
        ("w=0", [ids[i] for i in np.where(np.abs(w) <= ACTIVE_TOL)[0]]),
        ("w=u", [ids[i] for i in np.where(np.abs(w - problem.u_max) <= ACTIVE_TOL)[0]]),
        ("w=w0", [ids[i] for i in np.where(np.abs(w - problem.w0) <= ACTIVE_TOL)[0]]),
    ]
    parts = [f"{name}: {' '.join(v)}" for name, v in groups if v]
    if active:
        parts.append("turnover")
    return " | ".join(parts) or "-"


def analyze(problem: ALMProblem, T_max: float) -> TurnoverCase:
    """Barrera + SLSQP + diferencias finitas + conjunto activo + KKT para un T_max."""
    sol = solve_turnover(problem, T_max)
    w, x = sol.w, sol.x
    to = turnover(problem, w)
    active = abs(T_max - to) <= ACTIVE_TOL
    _, tau_bar, tau_newton = sol.tau_path[-1]
    f_minus = problem.f(solve_turnover(problem, T_max - FD_H).w)
    f_plus = problem.f(solve_turnover(problem, T_max + FD_H).w)
    sq = solve_slsqp_pq(problem, T_max, x)
    w_sq = to_w(problem, sq.x)
    return TurnoverCase(
        sol=sol, f=problem.f(w), gap=problem.duration_gap(w), turnover=to, sum_pq=float(x.sum()),
        active=active, tau_bar=tau_bar, tau_newton=tau_newton, tau_fd=(f_minus - f_plus) / (2 * FD_H),
        f_sq=problem.f(w_sq), dw_sq=float(np.linalg.norm(w - w_sq)), sq_success=bool(sq.success),
        sq_message=str(sq.message), labels=active_labels(problem, w, active),
        kkt=kkt_check_pq(problem, x) if active else kkt_check_w(problem, w),
    )


# ------------------------------------------------------------------------------------------------
# Salidas
# ------------------------------------------------------------------------------------------------

def trades(problem: ALMProblem, w: np.ndarray, tol: float = ACTIVE_TOL) -> tuple[str, str]:
    """('id +compra', ...) y ('id -venta', ...), ordenados por duración (ventas decreciente, compras creciente)."""
    d = w - problem.w0
    order = np.argsort(problem.D)
    buys = [f"{problem.ids[i]} +{d[i]:.4f}" for i in order if d[i] > tol]
    sells = [f"{problem.ids[i]} -{-d[i]:.4f}" + (" (todo)" if w[i] <= tol else "")
             for i in order[::-1] if d[i] < -tol]
    return ", ".join(buys) or "-", ", ".join(sells) or "-"


def sold_ids(problem: ALMProblem, w: np.ndarray, tol: float = ACTIVE_TOL) -> list[int]:
    """Índices de instrumentos vendidos (w_i < w0_i - tol)."""
    return [i for i in range(problem.n) if w[i] < problem.w0[i] - tol]


def check_sell_order(problem: ALMProblem, sols: dict[float, np.ndarray]) -> None:
    """
    (a) Para cada T_max, el conjunto vendido debe ser un prefijo de la lista ordenada por duración
        decreciente (todo vendido tiene mayor D que todo no vendido).
    (b) El primer T_max (rejilla fina) en que se vende cada instrumento de SELL_ORDER debe ser no
        decreciente a lo largo de SELL_ORDER.
    """
    rank = list(np.argsort(-problem.D))
    print("\nComprobación del orden de venta:")
    bad = []
    for T, w in sols.items():
        S = set(sold_ids(problem, w))
        prefix = set(rank[:len(S)])
        if S != prefix:
            bad.append((T, sorted(S - prefix), sorted(prefix - S)))
    if bad:
        print(f"  (a) DISCREPANCIA: en {len(bad)} valores de T_max lo vendido no es un prefijo por duración:")
        for T, extra, missing in bad:
            print(f"      T_max = {T:.2f}: vendidos fuera de orden "
                  f"{[problem.ids[i] for i in extra]}, no vendidos con más duración "
                  f"{[problem.ids[i] for i in missing]}")
    else:
        print(f"  (a) OK: en los {len(sols)} valores de T_max, lo vendido es siempre un prefijo de la "
              "lista por duración decreciente")

    first = {}
    for id_ in SELL_ORDER:
        i = problem.ids.index(id_)
        first[id_] = next((T for T, w in sols.items() if w[i] < problem.w0[i] - ACTIVE_TOL), None)
    print("  (b) Primer T_max en que se vende (rejilla de 0.01): "
          + ", ".join(f"{k} (D={problem.D[problem.ids.index(k)]:.2f}): "
                      + (f"{v:.2f}" if v is not None else "nunca") for k, v in first.items()))
    vals = [v if v is not None else np.inf for v in first.values()]
    ok = all(a <= b for a, b in zip(vals, vals[1:]))
    ties = [(a, b) for (a, va), (b, vb) in zip(list(first.items()), list(first.items())[1:]) if va == vb]
    print(f"      {'OK: orden no decreciente' if ok else 'DISCREPANCIA: el orden de entrada no sigue la duración'}"
          + (f" (empates a esta resolución: {ties})" if ties else ""))
    print(f"      Orden completo de entrada en ventas: "
          + ", ".join(f"{problem.ids[i]}@{next(T for T, w in sols.items() if w[i] < problem.w0[i] - ACTIVE_TOL):.2f}"
                      for i in rank if any(w[i] < problem.w0[i] - ACTIVE_TOL for w in sols.values())))


def plot_frontier(problem: ALMProblem, T_lp: np.ndarray, gap_lp: np.ndarray, T_qp: list[float],
                  gap_qp: np.ndarray, f_qp: np.ndarray, cases: list[TurnoverCase], T_min: float,
                  f_unc: float, path) -> None:
    """Desajuste |D^T w - D_L| y f frente a T_max: QP (barrera) con la frontera LP superpuesta."""
    import matplotlib.pyplot as plt
    from plots import BLUE, INK, MUTED, ORANGE

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    T_g = [c.sol.T_max for c in cases]

    ax1.plot(T_qp, np.abs(gap_qp), color=BLUE, label="QP con proximidad (barrera)")
    ax1.plot(T_g, [abs(c.gap) for c in cases], color=BLUE, marker="o", markersize=6, linestyle="none")
    ax1.plot(T_lp, gap_lp, color=ORANGE, linestyle="--", label=r"Frontera LP: $\min |D^\top w - D_L|$")
    ax1.set_ylabel(r"Desajuste residual $|D^\top w - D_L|$ (años)")
    ax1.set_title("Desajuste de duración", loc="left")
    ax1.set_ylim(bottom=0)

    pos = gap_lp > 1e-12
    ax2.plot(T_qp, f_qp, color=BLUE, label=r"$f^*(T_{\max})$ (barrera)")
    ax2.plot(T_g, [c.f for c in cases], color=BLUE, marker="o", markersize=6, linestyle="none")
    ax2.plot(T_lp[pos], gap_lp[pos]**2, color=ORANGE, linestyle="--",
             label=r"Cota inferior LP: $(\mathrm{desajuste}_{LP})^2$")
    ax2.axhline(f_unc, color=MUTED, linestyle=":", linewidth=1.5)
    ax2.annotate(f"sin límite: f = {f_unc:.4f}", xy=(T_lp[0], f_unc), xytext=(0, 4),
                 textcoords="offset points", ha="left", va="bottom", color=MUTED, fontsize=8)
    ax2.set_yscale("log")
    ax2.set_ylabel(r"$f(w^*)$ (escala log)")
    ax2.set_title("Función objetivo", loc="left")

    for ax in (ax1, ax2):
        for xv, txt, dx, ha in [(T_min, f"T_min = {T_min:.3f}", -3, "right"),
                                (TO_PEN_REF, f"sin límite\n{TO_PEN_REF:.4f}", 3, "left")]:
            ax.axvline(xv, color=MUTED, linestyle=":", linewidth=1)
            ax.annotate(txt, xy=(xv, 1.0), xycoords=("data", "axes fraction"), xytext=(dx, -3),
                        textcoords="offset points", ha=ha, va="top", color=MUTED, fontsize=8)
        ax.set_xlabel(r"$T_{\max}$")
        ax.legend(loc="lower left", fontsize=8)
    fig.suptitle("Frontera turnover–desajuste", x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_weights_turnover(problem: ALMProblem, w_04: np.ndarray, w_06: np.ndarray, w_unc: np.ndarray,
                          path) -> None:
    """Barras agrupadas: w0, T_max = 0.4, T_max = 0.6 y sin límite de turnover."""
    import matplotlib.pyplot as plt
    from plots import AQUA, BLUE, INK, MUTED, ORANGE

    YELLOW = "#eda100"  # slot 4 de la paleta categórica validada
    series = [
        (problem.w0, BLUE, "Cartera actual $w_0$", None),
        (w_04, ORANGE, r"$T_{\max} = 0.4$", None),
        (w_06, AQUA, r"$T_{\max} = 0.6$", "////"),
        (w_unc, YELLOW, f"Sin límite de turnover (penalización, turnover {turnover(problem, w_unc):.4f})",
         "\\\\\\\\"),
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
    ax.set_title("Pesos por instrumento con límite de turnover", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=2)
    ax.set_ylim(0, problem.u_max.max() + 0.06)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    from plots import FIG_DIR

    prob = load_problem(lam=0.1)
    n = prob.n
    print(f"Restricción de turnover (lam = {prob.lam}, n = {n}, variables x = (p, q) in R^{2 * n})")

    # ---------------- Parte 1: turnover mínimo ----------------
    x_min = min_turnover_lp(prob)
    w_min = to_w(prob, x_min)
    T_min = turnover(prob, w_min)
    print("\n[1] Turnover mínimo para cerrar el gap (LP):")
    print(f"  T_min = sum|w - w0| = {T_min:.6f}   (objetivo LP 1^T(p+q) = {x_min.sum():.6f})")
    print(f"  gap = {prob.duration_gap(w_min):+.2e},  sum(w) = {w_min.sum():.12f}")
    buys, sells = trades(prob, w_min)
    print(f"  Compras: {buys}")
    print(f"  Ventas:  {sells}")

    # ---------------- Parte 2: frontera LP ----------------
    T_front = [round(0.1 * k, 1) for k in range(1, int(T_min / 0.1) + 1)] + [T_min]
    print("\n[2] Frontera turnover-desajuste (LP: min |D^T w - D_L|):")
    print(f"  {'T_max':>8}{'|gap| (años)':>14}{'turnover':>11}{'D^T w':>9}  compras / ventas")
    for T in T_front:
        w = to_w(prob, frontier_lp(prob, T))
        b, s = trades(prob, w)
        print(f"  {T:>8.4f}{abs(prob.duration_gap(w)):>14.6f}{turnover(prob, w):>11.6f}"
              f"{prob.D @ w:>9.4f}  +[{b}]  -[{s}]")

    # ---------------- Partes 3 y 4: QP con barrera ----------------
    print(f"\n[3] QP con barrera logarítmica en (p, q): m = 4n + 1 = {4 * n + 1} desigualdades, "
          f"r = {RS[0]:g}, ..., {RS[-1]:g}")
    for T in T_GRID[:1]:
        x0 = initial_point(prob, T)
        print(f"  Punto inicial (T_max = {T}): p = q = delta = {x0[0]:.6f}  "
              f"(min(u-w0) = {(prob.u_max - prob.w0).min():.4f}, min(w0) = {prob.w0.min():.4f}, "
              f"T/(2n) = {T / (2 * n):.4f})")
    cases = [analyze(prob, T) for T in T_GRID]

    print(f"\n  {'T_max':>6}{'iter':>6}{'parada':>12}{'f(w)':>14}{'gap (años)':>12}{'turnover':>11}"
          f"{'1^T(p+q)':>11}{'activa':>8}")
    for c in cases:
        print(f"  {c.sol.T_max:>6.1f}{c.sol.n_inner:>6}{c.sol.stop:>12}{c.f:>14.10f}{c.gap:>+12.6f}"
              f"{c.turnover:>11.7f}{c.sum_pq:>11.7f}{'sí' if c.active else 'no':>8}")

    print(f"\n  Precio sombra del turnover tau = -df*/dT_max (r = {RS[-1]:g}; DF: diferencias finitas "
          f"centrales con h = {FD_H:g}):")
    print(f"  {'T_max':>6}{'tau = r/s':>14}{'|dif DF|':>10}{'tau = r/(s+Δs)':>16}{'|dif DF|':>10}{'tau DF':>14}")
    for c in cases:
        print(f"  {c.sol.T_max:>6.1f}{c.tau_bar:>14.8f}{abs(c.tau_bar - c.tau_fd):>10.1e}"
              f"{c.tau_newton:>16.8f}{abs(c.tau_newton - c.tau_fd):>10.1e}{c.tau_fd:>14.8f}")
    c02 = cases[0]
    print(f"  Evolución con r (T_max = {c02.sol.T_max}),  r: r/s | r/(s+Δs):")
    for r, a, b in c02.sol.tau_path:
        print(f"    {r:>7.0e}{a:>14.8f}{b:>14.8f}")

    print(f"\n  Validación con SLSQP en (p, q) (ftol = 1e-12, desde la solución de la barrera):")
    print(f"  {'T_max':>6}{'f barrera':>16}{'f SLSQP':>16}{'f_bar - f_SQ':>14}{'||w - w_SQ||':>14}"
          f"{'success':>9}  coincide (tol {CMP_TOL:g})")
    for c in cases:
        ok = c.dw_sq < CMP_TOL and abs(c.f - c.f_sq) < CMP_TOL
        print(f"  {c.sol.T_max:>6.1f}{c.f:>16.12f}{c.f_sq:>16.12f}{c.f - c.f_sq:>+14.2e}{c.dw_sq:>14.2e}"
              f"{'sí' if c.sq_success else 'no':>9}  {'sí' if ok else 'NO'}"
              + ("" if c.sq_success else f"   [SLSQP: {c.sq_message}]"))

    print("\n[4] Restricciones activas (tol 1e-6):")
    for c in cases:
        print(f"  T_max = {c.sol.T_max:.1f}:  {c.labels}")

    # Comprobación T_max = 1.0 frente a penalty.py
    c1 = next(c for c in cases if c.sol.T_max == 1.0)
    w_pen = solve_penalty(prob)[-1].w
    dw_pen = float(np.linalg.norm(c1.sol.w - w_pen))
    print(f"\n  T_max = 1.0 frente a la penalización de penalty.py (sin turnover, mu = 1e8):")
    print(f"    restricción activa: {'sí' if c1.active else 'no'}  (holgura real T_max - turnover = "
          f"{1.0 - c1.turnover:.6f};  holgura en (p,q) = {1.0 - c1.sum_pq:.6f})")
    print(f"    tau (r/s) = {c1.tau_bar:.2e},  tau (r/(s+Δs)) = {c1.tau_newton:.2e},  tau (DF) = {c1.tau_fd:.2e}")
    print(f"    turnover real = {c1.turnover:.6f} (penalización {turnover(prob, w_pen):.6f}, "
          f"referencia {TO_PEN_REF}),  1^T(p+q) = {c1.sum_pq:.6f}  <- sobreestima")
    print(f"    ||w_bar - w_pen|| = {dw_pen:.2e},  |f_bar - f_pen| = {abs(c1.f - prob.f(w_pen)):.2e}")
    checks_1 = {
        "turnover inactivo": not c1.active,
        "multiplicador ~0 (|tau_bar|, |tau_DF| < 1e-5)": abs(c1.tau_bar) < CMP_TOL and abs(c1.tau_fd) < CMP_TOL,
        "w coincide con penalización (tol 1e-5)": dw_pen < CMP_TOL,
        "turnover = 0.9552 (tol 1e-4 por el redondeo a 4 cifras)": abs(c1.turnover - TO_PEN_REF) < 1e-4,
    }
    for k, v in checks_1.items():
        print(f"    {'OK' if v else 'FALLA'}: {k}")

    # ---------------- Parte 5: KKT ----------------
    print("\n[5] Verificación KKT independiente (conjunto activo detectado + mínimos cuadrados):")
    print(f"  {'T_max':>6}{'espacio':>9}{'#activas':>10}{'nu (LS)':>13}{'nu (barr.)':>13}{'tau (LS)':>12}"
          f"{'tau (DF)':>12}{'min mult.':>12}{'residuo':>11}  ok")
    for c in cases:
        k = c.kkt
        print(f"  {c.sol.T_max:>6.1f}{k.space:>9}{k.n_active:>10}{k.nu:>+13.8f}{c.sol.nu:>+13.8f}"
              f"{k.tau:>12.8f}{c.tau_fd:>12.8f}{k.min_mult:>12.2e}{k.resid:>11.2e}  {'sí' if k.ok else 'NO'}")

    # ---------------- Orden de venta ----------------
    print("\nOrden de venta (ventas w0_i - w_i por duración decreciente; '(todo)' = w_i = 0):")
    for c in cases:
        _, s = trades(prob, c.sol.w)
        print(f"  T_max = {c.sol.T_max:.1f}:  {s}")
    print("\nCompras:")
    for c in cases:
        b, _ = trades(prob, c.sol.w)
        print(f"  T_max = {c.sol.T_max:.1f}:  {b}")

    fine = {T: solve_turnover(prob, T) for T in T_FINE}
    check_sell_order(prob, {T: s.w for T, s in fine.items()})
    i01 = prob.ids.index("A01")
    at_u = [c.sol.T_max for c in cases if abs(c.sol.w[i01] - prob.u_max[i01]) <= ACTIVE_TOL]
    print(f"  A01 (D = {prob.D[i01]:.3f}, u_max = {prob.u_max[i01]}): w_A01 por T_max = "
          + ", ".join(f"{c.sol.T_max:.1f}: {c.sol.w[i01]:.4f}" for c in cases))
    print(f"      En u_max para T_max en {at_u if at_u else 'ninguno'}"
          + (f"; deja de estarlo a partir de T_max = "
             f"{next((c.sol.T_max for c in cases if c.sol.T_max > max(at_u) ), None)}" if at_u else ""))

    # ---------------- Figuras ----------------
    T_lp = np.round(np.arange(0.0, 1.1001, 0.01), 2)
    gap_lp = np.array([abs(prob.duration_gap(to_w(prob, frontier_lp(prob, T)))) for T in T_lp])
    T_qp = list(fine)
    gap_qp = np.array([prob.duration_gap(s.w) for s in fine.values()])
    f_qp = np.array([prob.f(s.w) for s in fine.values()])

    FIG_DIR.mkdir(exist_ok=True)
    plot_frontier(prob, T_lp, gap_lp, T_qp, gap_qp, f_qp, cases, T_min, prob.f(w_pen),
                  FIG_DIR / "frontera_turnover.png")
    w_of = {c.sol.T_max: c.sol.w for c in cases}
    plot_weights_turnover(prob, w_of[0.4], w_of[0.6], w_pen, FIG_DIR / "pesos_turnover.png")
    print(f"\nFiguras: {FIG_DIR / 'frontera_turnover.png'}, {FIG_DIR / 'pesos_turnover.png'}")
