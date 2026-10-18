[← Volver al README](../README.md) · [Siguiente: Los datos →](02-datos.md)

# 1. El problema y su contexto

## El balance de un banco

Un banco tiene dos carteras. Los **activos** son lo que tiene: préstamos concedidos y bonos comprados. Los **pasivos** son lo que debe: depósitos de sus clientes y financiación obtenida en los mercados. Los dos lados generan flujos de caja futuros, y su valor actual depende de los tipos de interés.

Cuando los tipos suben, todo vale menos, pero no en la misma medida. Un bono a doce años pierde mucho valor; un préstamo a un año, casi nada. Si los activos del banco son más largos que sus deudas, una subida de tipos reduce más el valor de lo que tiene que el de lo que debe, y el banco pierde valor económico sin haber hecho nada.

## Dónde se gestiona

Este riesgo se llama **riesgo de tipo de interés en la cartera de inversión** (IRRBB, *Interest Rate Risk in the Banking Book*). En cada banco lo calcula el departamento de ALM (*Asset-Liability Management*) y riesgos, y el comité de activos y pasivos (ALCO) decide qué comprar y qué vender para controlarlo.

Además es obligatorio medirlo. El Comité de Supervisión Bancaria de Basilea exige calcular la variación del valor económico del banco,

$$\Delta EVE = \Delta V_A - \Delta V_L,$$

en seis escenarios estándar de movimiento de la curva de tipos. Esos escenarios se usan en la [fase 6](08-fase6-escenarios.md).

## La duración

La herramienta clásica para medir y controlar este riesgo es la duración.

Para un instrumento con flujos $CF_t$ y rentabilidad $y$, su precio es el valor presente de los flujos:

$$P = \sum_{t=1}^{T} \frac{CF_t}{(1+y)^t}.$$

La **duración de Macaulay** es el tiempo medio en que se recupera la inversión, ponderando cada instante por el peso de su flujo:

$$D_{\text{Mac}} = \frac{1}{P}\sum_{t=1}^{T} t\,\frac{CF_t}{(1+y)^t}.$$

La **duración modificada** mide cuánto cambia el precio cuando cambian los tipos:

$$D_{\text{mod}} = \frac{D_{\text{Mac}}}{1+y}, \qquad \frac{\Delta P}{P} \approx -D_{\text{mod}}\,\Delta y.$$

Una duración modificada de 9 significa que si los tipos suben un punto porcentual, el precio cae aproximadamente un 9 %. En el resto del proyecto, $D_i$ es la duración modificada del instrumento $i$.

### Ejemplo con dos instrumentos

| | A01 (préstamo a 1 año) | A02 (bono a 12 años) |
|---|---|---|
| Duración de Macaulay | 1,000 | 9,349 |
| Duración modificada | 0,976 | 9,004 |
| Caída de precio si los tipos suben 1 p.p. | ≈ 0,98 % | ≈ 9,0 % |

A01 tiene un único flujo, así que su duración coincide con el vencimiento. A02 paga cupones cada año, que adelantan la recuperación, y su duración (9,3 años) es menor que su vencimiento (12 años).

## La duración de una cartera

Si $w_i$ es el peso del instrumento $i$ en la cartera, la duración de la cartera es la media ponderada:

$$D_A(w) = \sum_{i=1}^{n} w_i D_i = D^\top w.$$

El **casamiento de duraciones** (*duration matching*) consiste en elegir los pesos $w$ para que $D_A(w)$ sea igual a la duración de los pasivos, $D_L$. Si se consigue, una subida o bajada de todos los tipos a la vez mueve igual los dos lados del balance y el valor económico apenas cambia.

En este proyecto, la cartera actual tiene una duración de 6,71 años y los pasivos de 3,24. Ese desajuste de 3,47 años es lo que hace que el banco pierda 66,5 M€ si los tipos suben 225 puntos básicos.

---

[← Volver al README](../README.md) · [Siguiente: Los datos →](02-datos.md)
