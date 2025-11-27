# wave simulation tutorial

> You can this simulator code from
> https://github.com/caoqiming/wave-simulator

## 代码结构介绍

- boundary_conditions.py 里实现了三种简单的边界条件
  BoundaryCondition 抽象类，定义了边界条件的接口。每一个边界条件都需要实现 1d 和 2d 仿真的 apply 函数。根据对称性，我们只需要实现一个方向的边界即可。这里都实现左边界。
  具体实现了 FixedBoundary，NeumannBoundary，UnlimitedBoundary
- boundary.py 边界相关的逻辑
  Boundary 类代表一个边界，包含如何利用对称性应用边界的逻辑
  Boundaries 类代表一组边界，包括多个边，以及边界的内部范围
- source.py 定义了波动源
  LineSource 类实现了一个线段形状的源
- oned_simulator.py 一维仿真
- twod_simulator.py 二维仿真

## 双缝干涉实验

设：

- 波长：λ
- 双缝中心间距（缝间距）：d
- 屏幕距离（从缝到观察屏的距离）：L
- 屏幕上距中心的垂直坐标：y
- 观察角：θ（缝平面法线与观察方向的夹角）

![image](./Optical%20path%20difference.jpeg)

光程差 $\Delta$ 的推导

P 点光程差 $\Delta = r_2 - r_1$

$$(1) \quad y_1 = y - \frac{d}{2}$$

$$(2) \quad y_2 = y + \frac{d}{2}$$

$$(3) \quad r_1^2 = L^2 + y_1^2$$

$$(4) \quad r_2^2 = L^2 + y_2^2$$

重点来了,当 $L \gg d, y$ 时

$$(5) \quad r_2 + r_1 \approx 2L$$

由 (4)-(3) 式得：
$$r_2^2 - r_1^2 = y_2^2 - y_1^2$$

利用平方差公式，并代入 (5) 式的近似值：
$$(r_2 + r_1)(r_2 - r_1) = 2D \Delta$$

另一方面，代入 (1) 和 (2) 式：
$$y_2^2 - y_1^2 = (y_2 - y_1)(y_2 + y_1)$$
$$y_2 - y_1 = \left(y + \frac{d}{2}\right) - \left(y - \frac{d}{2}\right) = d$$
$$y_2 + y_1 = \left(y + \frac{d}{2}\right) + \left(y - \frac{d}{2}\right) = 2y$$
所以：
$$r_2^2 - r_1^2 = (d)(2y) = 2yd$$

联立两个结果：
$$2D \Delta = 2yd$$

$$\Delta = \frac{yd}{L}$$

几何与干涉条件：

- 亮条纹（相长干涉）条件：$Δ = m λ，m ∈ Z$。
- 远场近似（夫琅禾费衍射）且小角度：$sinθ ≈ tanθ ≈ y/L$

于是亮条纹的纵向位置

$$
y_m ≈ L · sinθ ≈ L · (m λ / d) = m · (L λ / d)
$$

相邻亮条纹间距（m 与 m+1 之差）为

$$
Δy = y\_{m+1} − y_m ≈ L λ / d
$$
