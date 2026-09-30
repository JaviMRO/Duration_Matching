"""
generate_data.py
-----------------
Genera el dataset sintetico del proyecto de ALM (duration matching):

1. Una cartera de ACTIVOS (bonos/prestamos con cupon fijo, pago anual, bullet
   al vencimiento) con su duracion de Macaulay y duracion modificada.
2. Una estructura simplificada de PASIVOS en bloques, con su duracion
   ponderada objetivo D_L.
3. Los parametros auxiliares que usara el optimizador mas adelante:
   pesos actuales w0_i y limites por instrumento u_i.

Todos los datos son sinteticos pero con ordenes de magnitud realistas para
una cartera de banca (notionales en millones de euros, cupones y tires en
tanto por uno).

Salidas (en ../data/):
    assets.csv       -> un instrumento por fila
    liabilities.csv  -> un bloque de pasivo por fila
    summary.json     -> D_L, duracion actual de la cartera, gap inicial
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Reproducibilidad: misma "cartera sintetica" cada vez que se ejecuta
RNG = np.random.default_rng(seed=42)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def macaulay_duration(notional: float, coupon_rate: float, maturity_years: int,
                       yield_rate: float) -> tuple[float, float]:
    """
    Duracion de Macaulay y duracion modificada de un bono/prestamo con
    cupon fijo anual y devolucion del principal (bullet) al vencimiento.

    D_Macaulay = ( sum_t t * CF_t / (1+y)^t ) / Precio
    D_Modificada = D_Macaulay / (1 + y)

    Se asume periodicidad anual (frecuencia = 1) por simplicidad, algo
    razonable para un proyecto de este alcance.
    """
    t = np.arange(1, maturity_years + 1)
    coupon = notional * coupon_rate
    cash_flows = np.full(maturity_years, coupon, dtype=float)
    cash_flows[-1] += notional  # ultimo flujo incluye el principal

    discount_factors = 1.0 / (1.0 + yield_rate) ** t
    pv_cash_flows = cash_flows * discount_factors
    price = pv_cash_flows.sum()

    macaulay = (t * pv_cash_flows).sum() / price
    modified = macaulay / (1.0 + yield_rate)
    return macaulay, modified, price


def build_asset_portfolio(n_instruments: int = 15) -> pd.DataFrame:
    """
    Construye n instrumentos de renta fija con caracteristicas variadas:
    - Notional entre 10 y 100 millones de euros
    - Cupon entre 1.5% y 5.5%
    - Vencimiento entre 1 y 15 años (mezcla de corto, medio y largo plazo)
    - Yield (tir de mercado) entre 2% y 4.5%, algo correlacionada con el plazo
    """
    rows = []
    for i in range(n_instruments):
        maturity = int(RNG.choice([1, 2, 3, 5, 7, 10, 12, 15]))
        notional = round(RNG.uniform(10, 100), 1)
        coupon_rate = round(RNG.uniform(0.015, 0.055), 4)
        # Yield con ligera dependencia del plazo (curva ascendente + ruido)
        base_yield = 0.02 + 0.0015 * maturity
        yield_rate = round(base_yield + RNG.normal(0, 0.003), 4)

        macaulay, modified, price = macaulay_duration(
            notional, coupon_rate, maturity, yield_rate
        )

        rows.append({
            "instrument_id": f"A{i+1:02d}",
            "tipo": "prestamo" if maturity <= 5 else "bono",
            "notional_mm_eur": notional,
            "cupon": coupon_rate,
            "vencimiento_años": maturity,
            "yield_mercado": yield_rate,
            "precio_mm_eur": round(price, 3),
            "duracion_macaulay": round(macaulay, 3),
            "duracion_modificada": round(modified, 3),
        })

    df = pd.DataFrame(rows)

    # Peso actual w0_i = valor de mercado del instrumento / valor total cartera
    df["w0_actual"] = df["precio_mm_eur"] / df["precio_mm_eur"].sum()

    # Limite por instrumento u_i: ningun instrumento puede pesar mas de un
    # 20% de la cartera (regla tipica de diversificacion/concentracion)
    df["u_max"] = 0.20

    return df


def build_liability_structure() -> pd.DataFrame:
    """
    Estructura de pasivos simplificada en 3 bloques, tipica de un balance
    bancario minorista:

    - Depositos "core" (comportamiento pegajoso -> duracion larga aunque
      sean depositos a la vista, por la literatura de IRRBB)
    - Depositos a plazo (duracion contractual media)
    - Financiacion mayorista (duracion corta, contractual)
    """
    liabilities = pd.DataFrame([
        {"bloque": "depositos_core", "notional_mm_eur": 600.0, "duracion": 4.5},
        {"bloque": "depositos_plazo", "notional_mm_eur": 250.0, "duracion": 1.8},
        {"bloque": "financiacion_mayorista", "notional_mm_eur": 150.0, "duracion": 0.6},
    ])
    liabilities["peso"] = liabilities["notional_mm_eur"] / liabilities["notional_mm_eur"].sum()
    return liabilities


def main():
    assets = build_asset_portfolio(n_instruments=15)
    liabilities = build_liability_structure()

    # Duracion objetivo de los pasivos: media ponderada por notional
    D_L = float((liabilities["peso"] * liabilities["duracion"]).sum())

    # Duracion actual de la cartera de activos (con los pesos w0 actuales)
    D_actual = float((assets["w0_actual"] * assets["duracion_modificada"]).sum())

    gap_inicial = D_actual - D_L

    assets.to_csv(DATA_DIR / "assets.csv", index=False)
    liabilities.to_csv(DATA_DIR / "liabilities.csv", index=False)

    summary = {
        "n_instrumentos": len(assets),
        "duracion_objetivo_pasivos_D_L": round(D_L, 4),
        "duracion_actual_cartera_D_actual": round(D_actual, 4),
        "gap_inicial_años": round(gap_inicial, 4),
        "nota": "gap = D_actual - D_L. Positivo => activos mas sensibles a "
                "tipos que pasivos; el optimizador debera reducirlo hacia 0.",
    }
    with open(DATA_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=== Cartera de activos (primeras filas) ===")
    print(assets.head().to_string(index=False))
    print("\n=== Estructura de pasivos ===")
    print(liabilities.to_string(index=False))
    print("\n=== Resumen ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
