# First-order One-way Wave Equation

一阶单向波动方程（First-order One-way Wave Equation）是从经典的**二阶波动方程**推导出来的，它描述了波在一个方向上的独立传播。

### 推导步骤

#### 1. 从二阶波动方程开始

我们从 1 维空间中的经典齐次波动方程开始：
$$\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}$$
其中：

- $u(x, t)$ 是波的位移或场量（例如声压、电场等）。
- $t$ 是时间。
- $x$ 是空间坐标。
- $c$ 是波的传播速度。

#### 2. 使用 D'Alembert 解

二阶波动方程的通解由 D'Alembert 给出：
$$u(x, t) = f(x - ct) + g(x + ct)$$
其中：

- $f(x - ct)$ 表示**向右传播**的波（正 $x$ 方向）。
- $g(x + ct)$ 表示**向左传播**的波（负 $x$ 方向）。
- $f$ 和 $g$ 是任意可微函数，由初始条件确定。

#### 3. 分离单向波

我们现在分别考虑这两种波：

**A. 向右传播的波 (Right-traveling wave):**
假设我们的波场中**只存在向右传播的波**，即 $u(x, t) = f(x - ct)$，且 $g(x+ct) = 0$。

现在我们对 $u$ 求关于 $t$ 和 $x$ 的一阶偏导数：
$$\frac{\partial u}{\partial t} = \frac{\partial f}{\partial (x - ct)} \cdot \frac{\partial (x - ct)}{\partial t} = f' \cdot (-c) = -c f'$$
$$\frac{\partial u}{\partial x} = \frac{\partial f}{\partial (x - ct)} \cdot \frac{\partial (x - ct)}{\partial x} = f' \cdot (1) = f'$$
将两个结果联立：
$$\frac{\partial u}{\partial t} = -c \left( \frac{\partial u}{\partial x} \right)$$
移项后，即可得到**向右传播波的一阶单向波动方程**：
$$\frac{\partial u}{\partial t} + c \frac{\partial u}{\partial x} = 0$$

---

**B. 向左传播的波 (Left-traveling wave):**
假设我们的波场中**只存在向左传播的波**，即 $u(x, t) = g(x + ct)$，且 $f(x-ct) = 0$。

我们对 $u$ 求关于 $t$ 和 $x$ 的一阶偏导数：
$$\frac{\partial u}{\partial t} = \frac{\partial g}{\partial (x + ct)} \cdot \frac{\partial (x + ct)}{\partial t} = g' \cdot (c) = c g'$$
$$\frac{\partial u}{\partial x} = \frac{\partial g}{\partial (x + ct)} \cdot \frac{\partial (x + ct)}{\partial x} = g' \cdot (1) = g'$$
将两个结果联立：
$$\frac{\partial u}{\partial t} = c \left( \frac{\partial u}{\partial x} \right)$$
移项后，即可得到**向左传播波的一阶单向波动方程**：
$$\frac{\partial u}{\partial t} - c \frac{\partial u}{\partial x} = 0$$

### 应用于边界条件（吸收边界）

这种一阶单向波动方程的意义在于，它**只描述特定方向上的波传播**。

当我们将一个有限的计算域的**边界**设置成这种形式时，我们实际上是**假设**在该边界处，**只存在从计算域内部向外传播的波**，而**没有从外部反射回来的波**。

例如，在**右边界 $x=L$** 处：

- 我们希望**向右传播的波** $\left(\frac{\partial u}{\partial t} + c \frac{\partial u}{\partial x} = 0\right)$ 能够顺利通过边界。
- 我们不希望有**向左传播的波** $\left(\frac{\partial u}{\partial t} - c \frac{\partial u}{\partial x} = 0\right)$ 出现（即没有反射）。

因此，我们选择 **$\frac{\partial u}{\partial t} + c \frac{\partial u}{\partial x} = 0$** 作为 $x=L$ 处的**吸收边界条件**，因为它强制边界处的波行为符合一个只向外传播的单向波。这就模拟出了该边界外侧为无限空间的效果。
