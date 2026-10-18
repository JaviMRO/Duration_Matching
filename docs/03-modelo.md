[← Anterior: Los datos](02-datos.md) · [README](../README.md) · [Siguiente: Fase 1 →](04-fase1-sin-restricciones.md)

# 3. El modelo matemático

## Variables

Las variables de decisión son los 15 pesos de la cartera,

$$w = (w_1, \dots, w_{15}) \in \mathbb{R}^{15},$$

donde $w_i$ es la parte del dinero invertida en el instrumento $i$.

## Función objetivo

La primera idea sería minimizar el desajuste al cuadrado, $(D^\top w - D_L)^2$. Pero esa función solo depende de la combinación $D^\top w$: tiene infinitos mínimos (todo el hiperplano $D^\top w = D_L$) y su Hessiana, $2DD^\top$, tiene rango uno, así que Newton no se puede aplicar.

Por eso se añade un término de proximidad a la cartera actual:

$$f(w) = \left(D^\top w - D_L\right)^2 + \lambda\,\lVert w - w_0 \rVert^2, \qquad \lambda = 0{,}1.$$

El término tiene dos lecturas. Matemáticamente, hace que la Hessiana sea definida positiva y que la solución sea única. Financieramente, dice que el banco quiere corregir la duración moviendo lo menos posible su cartera.

## Gradiente y Hessiana

$$\nabla f(w) = 2\left(D^\top w - D_L\right) D + 2\lambda\,(w - w_0),$$

$$\nabla^2 f(w) = H = 2\left(DD^\top + \lambda I\right).$$

La Hessiana es constante. Sus autovalores son $2\lambda$ (con multiplicidad $n-1$) y $2(\lVert D\rVert^2 + \lambda)$ (en la dirección de $D$), así que su número de condición es

$$\kappa(H) = 1 + \frac{\lVert D \rVert^2}{\lambda}.$$

Con los datos, $\lVert D\rVert^2 \approx 794$ y $\kappa \approx 7.944$: el problema está mal condicionado, lo que se nota mucho en la [fase 1](04-fase1-sin-restricciones.md).

Antes de usar el gradiente y la Hessiana en ningún método, se comprueban contra diferencias finitas centrales en puntos aleatorios, con un error relativo menor que $10^{-5}$ (módulo `objective.py`).

## Restricciones

| Regla del banco | Restricción | Se introduce en |
|---|---|---|
| Invertir el 100 % del dinero | $\mathbf{1}^\top w = 1$ | [Fase 2](05-fase2-lagrange.md) |
| No vender en corto | $w_i \ge 0$ | [Fases 3 y 4](06-fase3-4-penalizacion-barrera.md) |
| No pasar del 20 % en un instrumento | $w_i \le u_i = 0{,}20$ | [Fases 3 y 4](06-fase3-4-penalizacion-barrera.md) |
| Mover solo una parte de la cartera | $\sum_i \lvert w_i - w_{0,i} \rvert \le T_{\max}$ | [Fase 5](07-fase5-turnover.md) |

El problema completo es

$$\min_w \; f(w) \quad \text{s.a.} \quad \mathbf{1}^\top w = 1, \quad 0 \le w \le u, \quad \sum_i \lvert w_i - w_{0,i} \rvert \le T_{\max}.$$

El proyecto no lo resuelve de golpe: añade las restricciones de una en una, y cada vez que una restricción nueva rompe el método anterior se pasa a uno más general.

## Comprobación de factibilidad

Antes de imponer las restricciones conviene saber si el objetivo es alcanzable. Para eso se resuelven dos programas lineales (módulo `feasibility.py`):

$$D_{\min} = \min_w D^\top w, \qquad D_{\max} = \max_w D^\top w, \qquad \text{s.a. } \mathbf{1}^\top w = 1,\; 0 \le w \le u.$$

Como el conjunto factible es convexo y $D^\top w$ es continua, el desajuste puede anularse si y solo si $D_{\min} \le D_L \le D_{\max}$.

Se obtiene $D_{\min} = 2{,}434$ y $D_{\max} = 9{,}686$ años, así que $D_L = 3{,}24$ es alcanzable, aunque bastante cerca del extremo inferior. La cartera de duración mínima pone el 20 % en los cinco instrumentos más cortos (A01, A06, A08, A10 y A12) y nada en el resto.

---

[← Anterior: Los datos](02-datos.md) · [README](../README.md) · [Siguiente: Fase 1 →](04-fase1-sin-restricciones.md)
