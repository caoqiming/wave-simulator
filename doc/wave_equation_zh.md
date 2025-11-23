### 连续形式的非均匀介质波动方程（一维）

我们使用的连续形式是：

$$
\frac{\partial^2 A}{\partial t^2} = \frac{\partial}{\partial x} \left[ c^2(x) \frac{\partial A}{\partial x} \right]
$$

其中 $A$ 是波场，$c(x)$ 是随位置变化的波速。

### 有限差分近似

我们采用标准符号：

- $A_{i}^{j}$: 波场在空间点 $x_i$、时间 $t_j$ 的值。
- $c_{i}$: 波速在空间点 $x_i$ 的值（随位置变化，不随时间变化）。
- $\Delta t$: 时间步长。
- $\Delta x$: 空间步长。

#### 1. 时间二阶导数 $\frac{\partial^2 A}{\partial t^2}$ 的近似

使用中心差分：

$$
\frac{\partial^2 A}{\partial t^2} \approx \frac{A_{i}^{j+1} - 2A_{i}^{j} + A_{i}^{j-1}}{\Delta t^2}
$$

#### 2. 空间项 $\frac{\partial}{\partial x} \left[ c^2(x) \frac{\partial A}{\partial x} \right]$ 的近似

对包含 $c^2(x)$ 的项，我们使用两步中心差分来近似：

**第一步：近似中括号内的 $\left[ c^2(x) \frac{\partial A}{\partial x} \right]$**

我们首先需要定义 **$c^2(x) \frac{\partial A}{\partial x}$ 在半网格点 $x_{i+1/2}$ 处的值**。

在 $x_{i+1/2}$ 处：

$$
\left[ c^2 \frac{\partial A}{\partial x} \right]_{i+1/2}^{j} \approx (c^2)_{i+1/2} \frac{A_{i+1}^{j} - A_{i}^{j}}{\Delta x}
$$

为了计算 $(c^2)_{i+1/2}$，通常使用相邻网格点 $c_{i}$ 和 $c_{i+1}$ 的**平均值**（或简单地使用它们平方的平均）：

$$
(c^2)_{i+1/2} \approx \frac{c_{i+1}^2 + c_{i}^2}{2}
$$

因此，第一步近似为：

$$
F_{i+1/2}^j = \frac{c_{i+1}^2 + c_{i}^2}{2} \frac{A_{i+1}^{j} - A_{i}^{j}}{\Delta x}
$$

**第二步：近似外面的 $\frac{\partial}{\partial x}$**

现在，我们对第一步得到的结果（可以看作一个通量 $F$）使用中心差分：

$$
\frac{\partial}{\partial x} \left[ c^2 \frac{\partial A}{\partial x} \right] \approx \frac{F_{i+1/2}^j - F_{i-1/2}^j}{\Delta x}
$$

其中 $F_{i-1/2}^j$ 为：
$$F_{i-1/2}^j = \frac{c_{i}^2 + c_{i-1}^2}{2} \frac{A_{i}^{j} - A_{i-1}^{j}}{\Delta x}$$

将 $F_{i+1/2}^j$ 和 $F_{i-1/2}^j$ 代入，得到空间项的最终近似：

$$
\frac{\partial}{\partial x} \left[ c^2 \frac{\partial A}{\partial x} \right] \approx \frac{1}{\Delta x^2} \left[ \frac{c_{i+1}^2 + c_{i}^2}{2} (A_{i+1}^{j} - A_{i}^{j}) - \frac{c_{i}^2 + c_{i-1}^2}{2} (A_{i}^{j} - A_{i-1}^{j}) \right]
$$

### 3. 完整的有限差分方程 (FDM Formula)

将时间近似和空间近似代入连续方程 $\frac{\partial^2 A}{\partial t^2} = \frac{\partial}{\partial x} \left[ c^2 \frac{\partial A}{\partial x} \right]$：

$$\frac{A_{i}^{j+1} - 2A_{i}^{j} + A_{i}^{j-1}}{\Delta t^2} = \frac{1}{\Delta x^2} \left[ \frac{c_{i+1}^2 + c_{i}^2}{2} (A_{i+1}^{j} - A_{i}^{j}) - \frac{c_{i}^2 + c_{i-1}^2}{2} (A_{i}^{j} - A_{i-1}^{j}) \right]$$

**求解 $A_{i}^{j+1}$（时间步进形式）：**

令 $\lambda = \left(\frac{\Delta t}{\Delta x}\right)^2$，整理得到最终的显式有限差分公式：

$$
A_{i}^{j+1} = 2A_{i}^{j} - A_{i}^{j-1} + \frac{\lambda}{2} \left[ (c_{i+1}^2 + c_{i}^2) (A_{i+1}^{j} - A_{i}^{j}) - (c_{i}^2 + c_{i-1}^2) (A_{i}^{j} - A_{i-1}^{j}) \right]
$$
