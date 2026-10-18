[← Anterior: El problema](01-contexto.md) · [README](../README.md) · [Siguiente: El modelo →](03-modelo.md)

# 2. Los datos

Al no disponer de datos reales de un banco, se genera un conjunto sintético con órdenes de magnitud realistas. El script es [`data/generate_data.py`](../data/generate_data.py) y usa una semilla fija (`seed=42`), así que cada ejecución produce exactamente los mismos datos.

```bash
python data/generate_data.py
```

## Activos: `data/assets.csv`

Son 15 instrumentos de renta fija con cupón anual y devolución del principal al vencimiento (*bullet*):

- nominal entre 10 y 100 M€,
- cupón entre el 1,5 % y el 5,5 %,
- vencimiento en {1, 2, 3, 5, 7, 10, 12, 15} años,
- rentabilidad de mercado que crece con el plazo, $y = 0{,}02 + 0{,}0015\,T + \varepsilon$ con $\varepsilon \sim N(0,\,0{,}003^2)$, para imitar una curva de tipos ascendente.

Para cada instrumento se calculan su precio y sus duraciones. El peso actual es su valor de mercado sobre el total, $w_{0,i} = P_i / \sum_j P_j$, y el límite de concentración es $u_i = 0{,}20$ para todos.

| Columna | Significado |
|---|---|
| `instrument_id` | Identificador, de A01 a A15 |
| `tipo` | `prestamo` o `bono` |
| `notional_mm_eur` | Nominal en millones de euros |
| `cupon` | Tipo de cupón anual, en tanto por uno |
| `vencimiento_años` | Años hasta el vencimiento |
| `yield_mercado` | Rentabilidad de mercado, en tanto por uno |
| `precio_mm_eur` | Valor presente de los flujos, en M€ |
| `duracion_macaulay` | Duración de Macaulay, en años |
| `duracion_modificada` | Duración modificada $D_i$ |
| `w0_actual` | Peso en la cartera actual |
| `u_max` | Peso máximo permitido (0,20) |

## Pasivos: `data/liabilities.csv`

Los pasivos se agrupan en tres bloques típicos de un banco minorista:

| Bloque | Nominal (M€) | Peso | Duración (años) |
|---|---|---|---|
| Depósitos core | 600 | 0,60 | 4,5 |
| Depósitos a plazo | 250 | 0,25 | 1,8 |
| Financiación mayorista | 150 | 0,15 | 0,6 |

Los depósitos core son depósitos a la vista, pero tienen una duración de comportamiento larga porque los clientes los mantienen durante años. La duración de los pasivos es la media ponderada:

$$D_L = 0{,}60 \cdot 4{,}5 + 0{,}25 \cdot 1{,}8 + 0{,}15 \cdot 0{,}6 = 3{,}24 \text{ años}.$$

## Resumen: `data/summary.json`

Guarda las tres cifras que definen el punto de partida:

| Dato | Valor |
|---|---|
| Duración de los pasivos, $D_L$ | 3,24 años |
| Duración de la cartera actual, $D^\top w_0$ | 6,7133 años |
| Desajuste inicial, $\text{gap}_0 = D^\top w_0 - D_L$ | 3,4733 años |

Un desajuste positivo significa que los activos son más sensibles a los tipos que los pasivos. El objetivo del proyecto es llevarlo a cero.

---

[← Anterior: El problema](01-contexto.md) · [README](../README.md) · [Siguiente: El modelo →](03-modelo.md)
