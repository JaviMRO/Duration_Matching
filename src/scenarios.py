"""
Riesgo de tipo de interés (ΔEVE) de la cartera actual y de las carteras optimizadas ante los seis
escenarios estándar de Basilea (BCBS, IRRBB), con revalorización completa de los flujos:

    ΔEVE = ΔV_A - ΔV_L

Curva base: y(t) = 0.02 + 0.0015 t (la de generate_data.py sin ruido), composición anual.

Activos: cada instrumento es un bono/préstamo bullet con cupón anual (notional, cupón, vencimiento,
yield_mercado de assets.csv). Precio base = flujos descontados a su propia yield y_i (coincide con
precio_mm_eur); precio perturbado: el flujo en t se descuenta a y_i + ΔR(t).

Pasivos: cada bloque de liabilities.csv es un cupón cero con valor presente igual a su nominal y
vencimiento t_k tal que su duración modificada t_k / (1 + y(t_k)) es la duración del bloque
(brentq). Flujo: N_k (1 + y(t_k))^t_k; perturbado se descuenta a y(t_k) + ΔR(t_k).

Tamaño: V_A = V_L = 1000 M€ y la posición en el instrumento i vale 1000 w_i M€. La cartera actual
vale en mercado ~879 M€, pero el problema de optimización iguala duraciones D_A = D_L, y eso solo
inmuniza el valor económico si los valores también son iguales: a primer orden
    ΔEVE ≈ -(D_A V_A - D_L V_L) ΔR,
que se anula con D_A = D_L solo si V_A = V_L. Con V_A != V_L habría que igualar las duraciones en
euros, D_A V_A = D_L V_L (es decir, D_L' = D_L V_L / V_A como objetivo de duración de los activos).
Con V_A = V_L = 1000 M€ el objetivo D_L = 3.24 es el correcto.

Escenarios (x = 4 años; se suprime el ±):
    ΔR_corto(t) = R_corto exp(-t/x),   ΔR_largo(t) = R_largo (1 - exp(-t/x))
    1 paralelo ↑: +R_par          2 paralelo ↓: -R_par
    3 corto ↑:    +ΔR_corto       4 corto ↓:    -ΔR_corto
    5 steepener:  -0.65 |ΔR_corto| + 0.9 |ΔR_largo|
    6 flattener:  +0.8 |ΔR_corto| - 0.6 |ΔR_largo|
Calibración principal (euro, recalibrada, vigente desde 2026): R_par = 225 pb, R_corto = 350 pb,
R_largo = 200 pb. Secundaria (2016): 200 / 250 / 100 pb. No se aplica suelo a los tipos negativos.

Aproximaciones de Taylor para los paralelos (V_A = V_L = 1000, D y C ponderados por valor):
    1er orden:  ΔEVE ≈ -(D_A - D_L) 1000 ΔR
    2º orden:   ΔEVE ≈ -(D_A - D_L) 1000 ΔR + ½ (C_A - C_L) 1000 ΔR²
con D_i = sum t CF_t (1+y)^-(t+1) / P y C_i = sum t (t+1) CF_t (1+y)^-(t+2) / P. Como un paralelo
desplaza todas las yields lo mismo, estas son exactamente las series de Taylor en ΔR del valor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from objective import DATA_DIR, ALMProblem, load_problem
from penalty import solve_slsqp, turnover
from turnover import min_turnover_lp, solve_turnover, to_w

CURVE = (0.02, 0.0015)          # y(t) = a + b t
V_TOTAL = 1000.0                # V_A = V_L (M€)
X_DECAY = 4.0
CALIBRATIONS = {
    "2026": {"par": 0.0225, "short": 0.035, "long": 0.020},   # principal
    "2016": {"par": 0.0200, "short": 0.025, "long": 0.010},
}
SCENARIOS = ["paralelo ↑", "paralelo ↓", "corto ↑", "corto ↓", "steepener", "flattener"]
BUCKETS = [("corto (t < 3)", 0.0, 3.0), ("medio (3 <= t < 7)", 3.0, 7.0), ("largo (t >= 7)", 7.0, np.inf)]
T_PORTFOLIOS = [0.2, 0.4, 0.6, 0.8]
T_CURVE = [round(0.05 * k, 2) for k in range(0, 21)]       # 0, 0.05, ..., 1.0 (figura frente a T_max)
PRICE_TOL = 5e-4                # precio_mm_eur está redondeado a 3 decimales
DUR_TOL = 5e-4                  # duracion_modificada redondeada a 3 decimales


def base_curve(t: np.ndarray | float) -> np.ndarray | float:
    """y(t) = 0.02 + 0.0015 t."""
    return CURVE[0] + CURVE[1] * np.asarray(t)


def shock(scenario: str, t: np.ndarray | float, cal: dict[str, float]) -> np.ndarray:
    """ΔR(t) del escenario BCBS con la calibración cal (en tanto por uno)."""
    t = np.asarray(t, dtype=float)
    short = cal["short"] * np.exp(-t / X_DECAY)
    long_ = cal["long"] * (1.0 - np.exp(-t / X_DECAY))
    return {
        "paralelo ↑": np.full_like(t, cal["par"]),
        "paralelo ↓": np.full_like(t, -cal["par"]),
        "corto ↑": short,
        "corto ↓": -short,
        "steepener": -0.65 * short + 0.9 * long_,
        "flattener": 0.8 * short - 0.6 * long_,
    }[scenario]


@dataclass
class Book:
    """Flujos de caja de un conjunto de posiciones: el flujo j pertenece a la posición idx[j]."""

    names: list[str]
    idx: np.ndarray     # posición de cada flujo
    t: np.ndarray       # instante del flujo (años)
    cf: np.ndarray      # importe del flujo (M€, escala de los datos)
    y: np.ndarray       # tipo al que se descuenta el flujo en el escenario base

    @property
    def n(self) -> int:
        return len(self.names)

    def pv_flows(self, dR: np.ndarray | None = None) -> np.ndarray:
        """Valor presente de cada flujo descontado a y + ΔR(t)."""
        rate = self.y if dR is None else self.y + dR
        return self.cf * (1.0 + rate) ** -self.t

    def prices(self, dR: np.ndarray | None = None) -> np.ndarray:
        """Valor de cada posición (suma de sus flujos descontados)."""
        return np.bincount(self.idx, self.pv_flows(dR), minlength=self.n)

    def durations(self) -> np.ndarray:
        """Duración modificada D = sum t CF (1+y)^-(t+1) / P."""
        return np.bincount(self.idx, self.t / (1 + self.y) * self.pv_flows(), minlength=self.n) / self.prices()

    def convexities(self) -> np.ndarray:
        """Convexidad C = sum t (t+1) CF (1+y)^-(t+2) / P."""
        w = self.t * (self.t + 1) / (1 + self.y) ** 2
        return np.bincount(self.idx, w * self.pv_flows(), minlength=self.n) / self.prices()


def load_assets() -> tuple[Book, pd.DataFrame]:
    """Flujos de los activos (cupón anual, bullet) a partir de assets.csv."""
    df = pd.read_csv(DATA_DIR / "assets.csv")
    idx, t, cf, y = [], [], [], []
    for i, row in df.iterrows():
        m = int(row["vencimiento_años"])
        flows = np.full(m, row["notional_mm_eur"] * row["cupon"])
        flows[-1] += row["notional_mm_eur"]
        idx += [i] * m
        t += list(range(1, m + 1))
        cf += list(flows)
        y += [row["yield_mercado"]] * m
    book = Book(names=df["instrument_id"].tolist(), idx=np.array(idx), t=np.array(t, dtype=float),
                cf=np.array(cf), y=np.array(y))
    return book, df


def liability_maturity(D: float) -> float:
    """t tal que t / (1 + y(t)) = D (duración modificada de un cupón cero), por brentq."""
    return brentq(lambda t: t / (1.0 + base_curve(t)) - D, 1e-8, 100.0, xtol=1e-14)


def load_liabilities() -> tuple[Book, pd.DataFrame]:
    """Un cupón cero por bloque, con VP = nominal y duración modificada = duración del bloque."""
    df = pd.read_csv(DATA_DIR / "liabilities.csv")
    t = np.array([liability_maturity(D) for D in df["duracion"]])
    y = base_curve(t)
    N = df["notional_mm_eur"].to_numpy(dtype=float)
    book = Book(names=df["bloque"].tolist(), idx=np.arange(len(df)), t=t, cf=N * (1 + y) ** t, y=y)
    return book, df


@dataclass
class ALMBooks:
    """Activos, pasivos y la escala V_A = V_L = V_TOTAL."""

    assets: Book
    liabs: Book
    P0: np.ndarray      # precios base de los activos
    L0: np.ndarray      # VP base de los bloques de pasivo
    D_A: np.ndarray     # duraciones modificadas exactas de los activos
    C_A: np.ndarray
    D_L: float          # duración modificada de los pasivos (ponderada por VP)
    C_L: float

    def units(self, w: np.ndarray) -> np.ndarray:
        """Unidades de cada instrumento para que su posición valga V_TOTAL w_i."""
        return V_TOTAL * w / self.P0

    def liab_scale(self) -> float:
        """Factor que lleva el VP total de los pasivos a V_TOTAL."""
        return V_TOTAL / self.L0.sum()

    def flow_deltas(self, w: np.ndarray, dR_fun) -> tuple[np.ndarray, np.ndarray]:
        """(ΔPV de cada flujo de activo en la cartera w, ΔPV de cada flujo de pasivo) en M€."""
        A, L = self.assets, self.liabs
        dA = self.units(w)[A.idx] * (A.pv_flows(dR_fun(A.t)) - A.pv_flows())
        dL = self.liab_scale() * (L.pv_flows(dR_fun(L.t)) - L.pv_flows())
        return dA, dL

    def delta_eve(self, w: np.ndarray, scenario: str, cal: dict[str, float]) -> tuple[float, float, float]:
        """(ΔV_A, ΔV_L, ΔEVE) con revalorización completa."""
        dA, dL = self.flow_deltas(w, lambda t: shock(scenario, t, cal))
        return float(dA.sum()), float(dL.sum()), float(dA.sum() - dL.sum())

    def taylor(self, w: np.ndarray, dR: float) -> tuple[float, float]:
        """Aproximaciones de 1er y 2º orden de ΔEVE para un paralelo ΔR."""
        first = -(self.D_A @ w - self.D_L) * V_TOTAL * dR
        return first, first + 0.5 * (self.C_A @ w - self.C_L) * V_TOTAL * dR**2

    def min_rate(self, scenario: str, cal: dict[str, float]) -> tuple[float, str]:
        """Tipo mínimo al que se descuenta algún flujo (activos y pasivos) y dónde."""
        A, L = self.assets, self.liabs
        rA = A.y + shock(scenario, A.t, cal)
        rL = L.y + shock(scenario, L.t, cal)
        jA, jL = int(np.argmin(rA)), int(np.argmin(rL))
        if rA[jA] <= rL[jL]:
            return float(rA[jA]), f"{A.names[A.idx[jA]]}, t = {A.t[jA]:.0f}"
        return float(rL[jL]), f"{L.names[jL]}, t = {L.t[jL]:.2f}"


def build_books() -> tuple[ALMBooks, pd.DataFrame, pd.DataFrame]:
    assets, df_a = load_assets()
    liabs, df_l = load_liabilities()
    L0 = liabs.prices()
    return ALMBooks(
        assets=assets, liabs=liabs, P0=assets.prices(), L0=L0,
        D_A=assets.durations(), C_A=assets.convexities(),
        D_L=float(L0 @ liabs.durations() / L0.sum()), C_L=float(L0 @ liabs.convexities() / L0.sum()),
    ), df_a, df_l


def portfolios(prob: ALMProblem) -> dict[str, np.ndarray]:
    """Carteras a evaluar (en orden de presentación)."""
    out = {"w0 (actual)": prob.w0.copy(), "óptima sin límite": solve_slsqp(prob),
           "turnover mínimo (LP)": to_w(prob, min_turnover_lp(prob))}
    for T in T_PORTFOLIOS:
        out[f"T_max = {T}"] = solve_turnover(prob, T).w
    return out


def eve_table(books: ALMBooks, ports: dict[str, np.ndarray], cal: dict[str, float]) -> dict[str, np.ndarray]:
    """ΔEVE (M€) de cada cartera en los seis escenarios."""
    return {name: np.array([books.delta_eve(w, s, cal)[2] for s in SCENARIOS]) for name, w in ports.items()}


def print_eve_table(prob: ALMProblem, ports: dict[str, np.ndarray], table: dict[str, np.ndarray]) -> None:
    head = f"  {'cartera':<22}{'gap D':>8}{'turnover':>10}" + "".join(f"{s:>12}" for s in SCENARIOS)
    print(head + f"{'peor caso':>12}  escenario")
    for name, w in ports.items():
        v = table[name]
        k = int(np.argmin(v))
        print(f"  {name:<22}{prob.duration_gap(w):>+8.4f}{turnover(prob, w):>10.4f}"
              + "".join(f"{x:>+12.2f}" for x in v) + f"{v[k]:>+12.2f}  {SCENARIOS[k]}")


def bucket_of(t: np.ndarray) -> np.ndarray:
    """Índice de tramo de cada instante t."""
    return np.array([next(k for k, (_, lo, hi) in enumerate(BUCKETS) if lo <= x < hi) for x in t])


def print_buckets(books: ALMBooks, w: np.ndarray, cal: dict[str, float]) -> None:
    """VP, duración en euros (DV = sum t/(1+y) VP) y ΔEVE por tramo y escenario."""
    A, L = books.assets, books.liabs
    bA, bL = bucket_of(A.t), bucket_of(L.t)
    pvA = books.units(w)[A.idx] * A.pv_flows()
    pvL = books.liab_scale() * L.pv_flows()
    ddA, ddL = A.t / (1 + A.y) * pvA, L.t / (1 + L.y) * pvL
    nb = len(BUCKETS)
    agg = lambda b, v: np.bincount(b, v, minlength=nb)

    print(f"  {'tramo':<20}{'VP activos':>12}{'VP pasivos':>12}{'DV activos':>12}{'DV pasivos':>12}"
          f"{'DV neto':>10}")
    for k, (name, *_) in enumerate(BUCKETS):
        print(f"  {name:<20}{agg(bA, pvA)[k]:>12.2f}{agg(bL, pvL)[k]:>12.2f}{agg(bA, ddA)[k]:>12.2f}"
              f"{agg(bL, ddL)[k]:>12.2f}{agg(bA, ddA)[k] - agg(bL, ddL)[k]:>+10.2f}")
    print(f"  {'total':<20}{pvA.sum():>12.2f}{pvL.sum():>12.2f}{ddA.sum():>12.2f}{ddL.sum():>12.2f}"
          f"{ddA.sum() - ddL.sum():>+10.2f}")
    print("  (DV = duración en euros, M€·año: un paralelo de +1 pb cambia el valor del tramo en -DV·1e-4 M€)")

    print(f"\n  ΔEVE por tramo (M€) = ΔV_A(tramo) - ΔV_L(tramo):")
    print(f"  {'tramo':<20}" + "".join(f"{s:>12}" for s in SCENARIOS))
    tot = np.zeros(len(SCENARIOS))
    for k, (name, *_) in enumerate(BUCKETS):
        row = []
        for s in SCENARIOS:
            dA, dL = books.flow_deltas(w, lambda t, s=s: shock(s, t, cal))
            row.append(agg(bA, dA)[k] - agg(bL, dL)[k])
        tot += row
        print(f"  {name:<20}" + "".join(f"{x:>+12.2f}" for x in row))
    print(f"  {'total':<20}" + "".join(f"{x:>+12.2f}" for x in tot))

    # Aproximación lineal por tramo: -DV_neto(tramo) x ΔR medio del tramo (ponderado por DV)
    print(f"\n  ΔR medio por tramo (pb, ponderado por DV de activos y pasivos) en cada escenario:")
    print(f"  {'tramo':<20}" + "".join(f"{s:>12}" for s in SCENARIOS))
    for k, (name, *_) in enumerate(BUCKETS):
        mA, mL = bA == k, bL == k
        wts = np.concatenate([ddA[mA], ddL[mL]])
        ts = np.concatenate([A.t[mA], L.t[mL]])
        print(f"  {name:<20}" + "".join(f"{1e4 * np.average(shock(s, ts, cal), weights=wts):>12.1f}"
                                        for s in SCENARIOS))


def plot_curves(path: Path) -> None:
    """Los seis ΔR(t) (calibración principal) para t en [0, 15]."""
    import matplotlib.pyplot as plt
    from plots import AQUA, BLUE, INK, ORANGE

    t = np.linspace(0, 15, 301)
    cal = CALIBRATIONS["2026"]
    style = {"paralelo ↑": (BLUE, "-"), "paralelo ↓": (BLUE, "--"), "corto ↑": (ORANGE, "-"),
             "corto ↓": (ORANGE, "--"), "steepener": (AQUA, "-"), "flattener": (AQUA, "--")}
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for s in SCENARIOS:
        c, ls = style[s]
        y = 1e4 * shock(s, t, cal)
        ax.plot(t, y, color=c, linestyle=ls, label=s)
        dy = {"corto ↑": 5, "corto ↓": -5}.get(s, 0)   # separa las dos etiquetas que acaban cerca de 0
        ax.annotate(s, xy=(t[-1], y[-1]), xytext=(4, dy), textcoords="offset points", va="center",
                    fontsize=8, color=INK)
    ax.axhline(0, color=INK, linewidth=1)
    ax.set_xlim(0, 17.5)
    ax.set_xlabel("Plazo t (años)")
    ax.set_ylabel(r"$\Delta R(t)$ (pb)")
    ax.set_title("Escenarios BCBS (calibración 2026: paralelo 225, corto 350, largo 200 pb)", loc="left")
    ax.legend(loc="upper center", ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_eve_bars(tables: dict[str, np.ndarray], names: list[tuple[str, str]], path: Path) -> None:
    """Barras agrupadas de ΔEVE por escenario para las carteras de names [(clave, etiqueta)]."""
    import matplotlib.pyplot as plt
    from plots import AQUA, BLUE, INK, MUTED, ORANGE

    colors = [BLUE, ORANGE, AQUA]
    x = np.arange(len(SCENARIOS))
    width = 0.26
    fig, ax = plt.subplots(figsize=(11, 5))
    for j, ((key, label), color) in enumerate(zip(names, colors)):
        v = tables[key]
        bars = ax.bar(x + (j - 1) * width, v, width=width, color=color, label=label,
                      edgecolor="white", linewidth=1)
        for b, val in zip(bars, v):
            ax.annotate(f"{val:+.1f}", xy=(b.get_x() + b.get_width() / 2, val),
                        xytext=(0, 3 if val >= 0 else -3), textcoords="offset points",
                        ha="center", va="bottom" if val >= 0 else "top", fontsize=7, color=MUTED)
    ax.axhline(0, color=INK, linewidth=1)
    ax.set_xticks(x, SCENARIOS)
    ax.set_ylabel(r"$\Delta EVE$ (M€)")
    ax.set_title(r"$\Delta EVE$ por escenario (calibración 2026, $V_A = V_L = 1000$ M€)", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_eve_turnover(T: list[float], worst: dict[str, np.ndarray], per_scen: np.ndarray, gap: np.ndarray,
                      T_min: float, to_unc: float, path: Path) -> None:
    """
    Izquierda: peor caso de ΔEVE frente a T_max (dos calibraciones). Centro: ΔEVE de cada escenario
    (calibración 2026). Derecha: desajuste de duración |D^T w - D_L| (la frontera de turnover.py).
    """
    import matplotlib.pyplot as plt
    from plots import AQUA, BLUE, INK, MUTED, ORANGE

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))
    for (cal, v), c in zip(worst.items(), [BLUE, ORANGE]):
        ax1.plot(T, v, color=c, marker="o", markersize=5, label=f"Calibración {cal}")
    ax1.set_ylabel(r"Peor $\Delta EVE$ (M€)")
    ax1.set_title("Peor caso de los seis escenarios", loc="left")

    style = {"paralelo ↑": (BLUE, "-"), "paralelo ↓": (BLUE, "--"), "corto ↑": (ORANGE, "-"),
             "corto ↓": (ORANGE, "--"), "steepener": (AQUA, "-"), "flattener": (AQUA, "--")}
    for k, s in enumerate(SCENARIOS):
        c, ls = style[s]
        ax2.plot(T, per_scen[:, k], color=c, linestyle=ls, label=s)
    ax2.set_ylabel(r"$\Delta EVE$ (M€)")
    ax2.set_title("Por escenario (calibración 2026)", loc="left")

    ax3.plot(T, gap, color=BLUE, marker="o", markersize=5)
    ax3.set_ylabel(r"$|D^\top w - D_L|$ (años)")
    ax3.set_title("Desajuste de duración (QP)", loc="left")
    ax3.set_ylim(bottom=0)

    for ax, (yf, va) in [(ax1, (0.02, "bottom")), (ax2, (0.02, "bottom")), (ax3, (0.98, "top"))]:
        ax.axhline(0, color=INK, linewidth=1)
        for xv, txt, dx, ha in [(T_min, f"T_min = {T_min:.3f}", -3, "right"),
                                (to_unc, f"sin límite\n{to_unc:.4f}", 3, "left")]:
            ax.axvline(xv, color=MUTED, linestyle=":", linewidth=1)
            ax.annotate(txt, xy=(xv, yf), xycoords=("data", "axes fraction"), xytext=(dx, 0),
                        textcoords="offset points", ha=ha, va=va, color=MUTED, fontsize=8)
        ax.set_xlabel(r"$T_{\max}$")
    ax1.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8, ncol=2)
    fig.suptitle(r"Riesgo $\Delta EVE$ frente al límite de turnover", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    from plots import FIG_DIR

    prob = load_problem(lam=0.1)
    books, df_a, df_l = build_books()
    cal = CALIBRATIONS["2026"]

    # ---------------- Comprobaciones del modelo ----------------
    print("Modelo de valoración")
    dP = np.abs(books.P0 - df_a["precio_mm_eur"].to_numpy())
    dD = np.abs(books.D_A - df_a["duracion_modificada"].to_numpy())
    print(f"  Activos: max |precio - precio_mm_eur| = {dP.max():.2e} (tol {PRICE_TOL:g}),  "
          f"max |D - duracion_modificada| = {dD.max():.2e} (tol {DUR_TOL:g})")
    assert dP.max() < PRICE_TOL and dD.max() < DUR_TOL, "Los precios/duraciones no coinciden con assets.csv"
    print(f"  Valor de mercado de la cartera actual = {books.P0.sum():.3f} M€  ->  se reescala a "
          f"V_A = {V_TOTAL:.0f} M€ (posición i = 1000 w_i)")
    print("  Pasivos (cupón cero, VP = nominal, t_k / (1 + y(t_k)) = D_k):")
    print(f"    {'bloque':<24}{'VP':>8}{'D_k':>6}{'t_k':>10}{'y(t_k)':>9}{'D mod.':>9}{'convex.':>9}")
    DL_blocks, CL_blocks = books.liabs.durations(), books.liabs.convexities()
    for k, name in enumerate(books.liabs.names):
        print(f"    {name:<24}{books.L0[k]:>8.2f}{df_l['duracion'][k]:>6.2f}{books.liabs.t[k]:>10.6f}"
              f"{books.liabs.y[k]:>9.6f}{DL_blocks[k]:>9.6f}{CL_blocks[k]:>9.4f}")
    print(f"    D_L ponderada = {books.D_L:.10f}  (objetivo {prob.D_L}),  C_L = {books.C_L:.4f}")
    assert abs(books.D_L - prob.D_L) < 1e-10, "La duración de los pasivos no es 3.24"

    print("\nTipo mínimo de descuento resultante (sin suelo), calibración 2026 / 2016:")
    for s in SCENARIOS:
        (r1, w1), (r2, w2) = books.min_rate(s, cal), books.min_rate(s, CALIBRATIONS["2016"])
        print(f"  {s:<12}{1e4 * r1:>+8.1f} pb {'(' + w1 + ')':<30}{1e4 * r2:>+8.1f} pb ({w2})")

    # ---------------- 1. Tablas ΔEVE ----------------
    ports = portfolios(prob)
    tables = {}
    for name, c in CALIBRATIONS.items():
        tables[name] = eve_table(books, ports, c)
        tag = "principal: par 225 / corto 350 / largo 200 pb" if name == "2026" else \
            "2016: par 200 / corto 250 / largo 100 pb"
        print(f"\n[1] ΔEVE (M€), calibración {tag}; gap D = D^T w - D_L (años, D de assets.csv):")
        print_eve_table(prob, ports, tables[name])

    w_opt = ports["óptima sin límite"]
    v_opt = tables["2026"]["óptima sin límite"]
    v_w0 = tables["2026"]["w0 (actual)"]
    print("\n  Referencias del prototipo (calibración 2026):")
    for s, ref in [("paralelo ↑", -66.5), ("paralelo ↓", 92.5), ("steepener", -66.1), ("flattener", 54.3)]:
        v = v_w0[SCENARIOS.index(s)]
        print(f"    w0 {s:<12} {v:>+8.2f}  (ref. {ref:+.1f},  dif. {v - ref:+.2f})")
    print(f"    óptima: rango [{v_opt.min():+.2f}, {v_opt.max():+.2f}]  (ref. [-3.4, +4.2])")

    # ---------------- 2. Paralelos: Taylor ----------------
    print("\n[2] Paralelos: revalorización completa frente a Taylor (calibración 2026)")
    print(f"  {'cartera':<22}{'D_A-D_L':>9}{'C_A-C_L':>9}{'escenario':>12}{'completa':>11}{'1er ord.':>11}"
          f"{'error':>9}{'2º ord.':>11}{'error':>9}")
    for name, w in ports.items():
        for s, dR in [("paralelo ↑", cal["par"]), ("paralelo ↓", -cal["par"])]:
            full = books.delta_eve(w, s, cal)[2]
            first, second = books.taylor(w, dR)
            print(f"  {name:<22}{books.D_A @ w - books.D_L:>+9.4f}{books.C_A @ w - books.C_L:>+9.3f}{s:>12}"
                  f"{full:>+11.3f}{first:>+11.3f}{full - first:>+9.3f}{second:>+11.3f}{full - second:>+9.3f}")
    print("  (D y C exactas a partir de los flujos; las de assets.csv están redondeadas a 3 decimales)")

    # ---------------- 3. Descomposición por tramos ----------------
    print("\n[3] Cartera óptima sin límite: descomposición por tramos de vencimiento del flujo")
    print(f"  D_A - D_L = {books.D_A @ w_opt - books.D_L:+.2e} años,  C_A - C_L = {books.C_A @ w_opt - books.C_L:+.3f}")
    print_buckets(books, w_opt, cal)
    print("\n  Misma descomposición para w0:")
    print_buckets(books, prob.w0, cal)

    # ---------------- Figuras ----------------
    FIG_DIR.mkdir(exist_ok=True)
    plot_curves(FIG_DIR / "curvas_shock.png")
    plot_eve_bars(tables["2026"], [("w0 (actual)", "Cartera actual $w_0$"),
                                   ("óptima sin límite", "Óptima sin límite de turnover"),
                                   ("T_max = 0.4", r"$T_{\max} = 0.4$")], FIG_DIR / "delta_eve_escenarios.png")

    sols = [solve_turnover(prob, T).w if T > 0 else prob.w0 for T in T_CURVE]
    per_scen = np.array([[books.delta_eve(w, s, cal)[2] for s in SCENARIOS] for w in sols])
    worst = {name: np.array([min(books.delta_eve(w, s, c)[2] for s in SCENARIOS) for w in sols])
             for name, c in CALIBRATIONS.items()}
    gap = np.array([abs(prob.duration_gap(w)) for w in sols])
    T_min = turnover(prob, ports["turnover mínimo (LP)"])
    print("\nPeor ΔEVE frente a T_max (QP de turnover.py; T_max = 0 es w0):")
    print(f"  {'T_max':>6}{'|gap D|':>9}{'peor 2026':>11}{'escenario':>12}{'peor 2016':>11}")
    for i, T in enumerate(T_CURVE):
        k = int(np.argmin(per_scen[i]))
        print(f"  {T:>6.2f}{gap[i]:>9.4f}{worst['2026'][i]:>+11.2f}{SCENARIOS[k]:>12}{worst['2016'][i]:>+11.2f}")
    plot_eve_turnover(T_CURVE, worst, per_scen, gap, T_min, turnover(prob, w_opt),
                      FIG_DIR / "delta_eve_turnover.png")
    print(f"\nFiguras: {FIG_DIR / 'delta_eve_escenarios.png'}, {FIG_DIR / 'delta_eve_turnover.png'}, "
          f"{FIG_DIR / 'curvas_shock.png'}")
