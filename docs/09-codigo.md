[← Anterior: Fase 6](08-fase6-escenarios.md) · [README](../README.md) · [Siguiente: La presentación →](10-presentacion.md)

# 9. El código

Todo está en Python con NumPy, SciPy, pandas y Matplotlib. Cada módulo es ejecutable por separado, imprime sus resultados y se valida con comprobaciones automáticas (`assert`) al ejecutarse.

## Instalación

```bash
pip install -r requirements.txt
```

## Orden de ejecución

Los datos se generan desde la raíz del repositorio. El resto de módulos se importan entre sí, así que se ejecutan desde `src`:

```bash
python data/generate_data.py

cd src
python objective.py        # comprueba gradiente y Hessiana
python feasibility.py      # rango de duraciones alcanzable
python optimizers.py       # fase 1
python plots.py            # figuras de la fase 1 y 2
python lagrange.py         # fase 2
python penalty.py          # fase 3
python interior_point.py   # fase 4
python turnover.py         # fase 5
python scenarios.py        # fase 6
```

Las figuras se guardan en `figures/`.

## Módulos

| Módulo | Qué hace | Documentación | Figuras que genera |
|---|---|---|---|
| [`data/generate_data.py`](../data/generate_data.py) | Genera la cartera sintética, calcula precios y duraciones y escribe los CSV y el resumen | [Los datos](02-datos.md) | – |
| [`src/objective.py`](../src/objective.py) | Clase `ALMProblem`, carga de datos (`load_problem`), función objetivo, gradiente y Hessiana, y su validación con diferencias finitas | [El modelo](03-modelo.md) | – |
| [`src/feasibility.py`](../src/feasibility.py) | Dos programas lineales para obtener $D_{\min}$ y $D_{\max}$ | [El modelo](03-modelo.md) | – |
| [`src/optimizers.py`](../src/optimizers.py) | Gradiente (paso fijo y Armijo con la corrección de Hager-Zhang), Newton y solución cerrada | [Fase 1](04-fase1-sin-restricciones.md) | – |
| [`src/plots.py`](../src/plots.py) | Figuras de las fases 1 y 2 y la ruta común `FIG_DIR` | [Fase 1](04-fase1-sin-restricciones.md) | `convergencia.png`, `lambda_sweep.png`, `pesos_comparacion.png` |
| [`src/lagrange.py`](../src/lagrange.py) | Sistema KKT con la restricción de presupuesto y contraste con SLSQP | [Fase 2](05-fase2-lagrange.md) | – |
| [`src/penalty.py`](../src/penalty.py) | Penalización exterior cuadrática, multiplicadores y conjunto activo | [Fases 3 y 4](06-fase3-4-penalizacion-barrera.md) | `penalty_convergence.png`, `pesos_comparacion_v2.png` |
| [`src/interior_point.py`](../src/interior_point.py) | Barreras logarítmica e inversa con Newton con restricción de igualdad | [Fases 3 y 4](06-fase3-4-penalizacion-barrera.md) | `interior_point_convergence.png`, `trayectorias.png` |
| [`src/turnover.py`](../src/turnover.py) | Variables $(p, q)$, turnover mínimo, frontera, barrera con turnover y comprobaciones KKT | [Fase 5](07-fase5-turnover.md) | `frontera_turnover.png`, `pesos_turnover.png` |
| [`src/scenarios.py`](../src/scenarios.py) | Escenarios de Basilea, revalorización completa y $\Delta EVE$ por cartera | [Fase 6](08-fase6-escenarios.md) | `curvas_shock.png`, `delta_eve_escenarios.png`, `delta_eve_turnover.png` |

## Dependencias entre módulos

```
objective ─┬─ feasibility
           ├─ optimizers ─┬─ lagrange ── penalty ── interior_point ── turnover ── scenarios
           │              └─ plots
```

`objective.py` es la base: define el problema y lo carga desde `data/`. Cada fase reutiliza la anterior, por ejemplo la búsqueda de Armijo de `optimizers.py` o el Newton con restricción de igualdad de `interior_point.py`.

## Parámetros principales

| Parámetro | Valor | Dónde |
|---|---|---|
| Semilla de los datos | 42 | `generate_data.py` |
| λ (proximidad) | 0,1 | `load_problem(lam=0.1)` |
| Tolerancia de parada | $\lVert\nabla f\rVert < 10^{-8}$ | `optimizers.py` |
| Valores de μ (penalización) | 1, 10, …, 10⁸ | `MUS` en `penalty.py` |
| Valores de r (barrera) | 1, 10⁻¹, …, 10⁻¹⁰ | `RS` en `interior_point.py` |

---

[← Anterior: Fase 6](08-fase6-escenarios.md) · [README](../README.md) · [Siguiente: La presentación →](10-presentacion.md)
