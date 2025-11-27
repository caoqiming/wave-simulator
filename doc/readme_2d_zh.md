# 二维波仿真

## 波动方程

二维平面的波动方程（通常用于描述薄膜的振动，例如鼓面）是一个二阶线性偏微分方程。

假设介质密度 $\rho$ 是常数时，它的一般形式可以表示为：

$$\frac{\partial^2 u}{\partial t^2} = c^2(x, y) \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)$$

其中：

- $u(x, y, t)$ 表示在时间 $t$、位置 $(x, y)$ 处的**位移**（例如，薄膜偏离平衡位置的高度）。
- $t$ 是**时间**。
- $x$ 和 $y$ 是**空间坐标**。
- $c$ 是**波速**
- $\nabla^2$ 是二维拉普拉斯算子，定义为 $\nabla^2 = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}$。

## 差分形式

二维波动方程最常见的**显式有限差分形式**（通常使用中心差分进行二阶近似）如下所示。

我们定义一个离散化的网格，其中：

- $u_{i, j}^{k}$ 表示函数 $u$ 在空间点 $(x_i, y_j)$ 和时间 $t_k$ 处的值。
- $x_i = i \Delta x$
- $y_j = j \Delta y$
- $t_k = k \Delta t$
- $\Delta x$ 和 $\Delta y$ 是空间步长（通常取 $\Delta x = \Delta y = h$）。
- $\Delta t$ 是时间步长。

我们使用二阶中心差分来近似所有二阶偏导数：

1.  **时间二阶导数**：

    $$
    \frac{\partial^2 u}{\partial t^2} \approx \frac{u_{i, j}^{k+1} - 2 u_{i, j}^{k} + u_{i, j}^{k-1}}{(\Delta t)^2}
    $$

2.  **空间二阶导数（x 方向）**：

    $$
    \frac{\partial^2 u}{\partial x^2} \approx \frac{u_{i+1, j}^{k} - 2 u_{i, j}^{k} + u_{i-1, j}^{k}}{(\Delta x)^2}
    $$

3.  **空间二阶导数（y 方向）**：
    $$
    \frac{\partial^2 u}{\partial y^2} \approx \frac{u_{i, j+1}^{k} - 2 u_{i, j}^{k} + u_{i, j-1}^{k}}{(\Delta y)^2}
    $$

$$
\frac{u_{i, j}^{k+1} - 2 u_{i, j}^{k} + u_{i, j}^{k-1}}{(\Delta t)^2} = c_{i, j}^2 \left[ \frac{u_{i+1, j}^{k} - 2 u_{i, j}^{k} + u_{i-1, j}^{k}}{h^2} + \frac{u_{i, j+1}^{k} - 2 u_{i, j}^{k} + u_{i, j-1}^{k}}{h^2} \right]
$$

## 四阶近似

为了在空间上获得更高的精度，可将二维拉普拉斯算子用四阶中心差分近似。时间方向仍采用二阶中心差分（显式格式），这样在计算成本与稳定性之间平衡较好。

- 四阶空间中心差分（x 方向）：

  $$
  \frac{\partial^2 u}{\partial x^2}\Big|_{i,j} \approx \frac{-u_{i-2,j} + 16u_{i-1,j} - 30u_{i,j} + 16u_{i+1,j} - u_{i+2,j}}{12\,h^2}
  $$

- 四阶空间中心差分（y 方向）：

  $$
  \frac{\partial^2 u}{\partial y^2}\Big|_{i,j} \approx \frac{-u_{i,j-2} + 16u_{i,j-1} - 30u_{i,j} + 16u_{i,j+1} - u_{i,j+2}}{12\,h^2}
  $$

将上述近似代入二维波动方程，并保持时间二阶中心差分不变，可得显式更新公式（记 $c_{i,j} = c(x_i,y_j)$，且 $\Delta x=\Delta y=h$）：

$$
\frac{u_{i,j}^{k+1} - 2u_{i,j}^{k} + u_{i,j}^{k-1}}{(\Delta t)^2}
= c_{i,j}^2 \left[
\frac{-u_{i-2,j}^{k} + 16u_{i-1,j}^{k} - 30u_{i,j}^{k} + 16u_{i+1,j}^{k} - u_{i+2,j}^{k}}{12\,h^2}
+
\frac{-u_{i,j-2}^{k} + 16u_{i,j-1}^{k} - 30u_{i,j}^{k} + 16u_{i,j+1}^{k} - u_{i,j+2}^{k}}{12\,h^2}
\right]
$$

整理得到显式更新步骤：

$$
u_{i,j}^{k+1} = 2u_{i,j}^{k} - u_{i,j}^{k-1} + \left(\frac{\Delta t\, c_{i,j}}{h}\right)^2
$$

$$
\frac{- u_{i-2,j}^{k} + 16u_{i-1,j}^{k} - 30u_{i,j}^{k} + 16u_{i+1,j}^{k} - u_{i+2,j}^{k} - u_{i,j-2}^{k} + 16u_{i,j-1}^{k} - 30u_{i,j}^{k} + 16u_{i,j+1}^{k} - u_{i,j+2}^{k}}{12}
$$

- 边界与幽灵点：由于四阶格式使用到 ±2 的点，需在边界处设定充分的边界条件或幽灵点（如固定/自由边界、吸收边界），以保证算子可用。
- 稳定性提示（均匀介质、常数波速 $c$）：四阶空间+二阶时间的显式格式一般满足类似 CFL 约束。对规则网格可采用保守选择
  $$
  \frac{c\,\Delta t}{h} \lesssim \frac{1}{\sqrt{2}}
  $$
  实际稳定上限与具体边界及实现相关，建议通过数值实验微调。

备注：若需要时间方向四阶精度，可使用多步法或 Runge–Kutta–Nyström 等方法，但实现与稳定性分析更为复杂，常见做法是保持时间二阶、空间四阶以获得较好综合效果。
