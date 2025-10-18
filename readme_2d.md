# 2d simulator

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
    $$\frac{\partial^2 u}{\partial t^2} \approx \frac{u_{i, j}^{k+1} - 2 u_{i, j}^{k} + u_{i, j}^{k-1}}{(\Delta t)^2}$$

2.  **空间二阶导数（x 方向）**：
    $$\frac{\partial^2 u}{\partial x^2} \approx \frac{u_{i+1, j}^{k} - 2 u_{i, j}^{k} + u_{i-1, j}^{k}}{(\Delta x)^2}$$

3.  **空间二阶导数（y 方向）**：
    $$\frac{\partial^2 u}{\partial y^2} \approx \frac{u_{i, j+1}^{k} - 2 u_{i, j}^{k} + u_{i, j-1}^{k}}{(\Delta y)^2}$$

$$\frac{u_{i, j}^{k+1} - 2 u_{i, j}^{k} + u_{i, j}^{k-1}}{(\Delta t)^2} = c_{i, j}^2 \left[ \frac{u_{i+1, j}^{k} - 2 u_{i, j}^{k} + u_{i-1, j}^{k}}{h^2} + \frac{u_{i, j+1}^{k} - 2 u_{i, j}^{k} + u_{i, j-1}^{k}}{h^2} \right]$$
