# 2D Wave Simulation

## Wave Equation

The wave equation in a two-dimensional plane (typically used to describe the vibration of a thin membrane, such as a drumhead) is a second-order linear partial differential equation.

Assuming the medium density $\rho$ is constant, its general form can be expressed as:

$$\frac{\partial^2 u}{\partial t^2} = c^2(x, y) \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)$$

Where:

- $u(x, y, t)$ represents the **displacement** (e.g., the height of the membrane from the equilibrium position) at time $t$ and position $(x, y)$.
- $t$ is **time**.
- $x$ and $y$ are **spatial coordinates**.
- $c$ is the **wave speed**.
- $\nabla^2$ is the two-dimensional Laplacian operator, defined as $\nabla^2 = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}$.

## Finite Difference Form

The most common **explicit finite difference form** of the two-dimensional wave equation (usually using central difference for second-order approximation) is as follows.

We define a discretized grid, where:

- $u_{i, j}^{k}$ represents the value of the function $u$ at the spatial point $(x_i, y_j)$ and time $t_k$.
- $x_i = i \Delta x$
- $y_j = j \Delta y$
- $t_k = k \Delta t$
- $\Delta x$ and $\Delta y$ are the spatial step sizes (usually $\Delta x = \Delta y = h$).
- $\Delta t$ is the time step size.

We use the second-order central difference to approximate all second partial derivatives:

1.  **Second-order time derivative**:

    $$
    \frac{\partial^2 u}{\partial t^2} \approx \frac{u_{i, j}^{k+1} - 2 u_{i, j}^{k} + u_{i, j}^{k-1}}{(\Delta t)^2}
    $$

2.  **Second-order spatial derivative (x-direction)**:

    $$
    \frac{\partial^2 u}{\partial x^2} \approx \frac{u_{i+1, j}^{k} - 2 u_{i, j}^{k} + u_{i-1, j}^{k}}{(\Delta x)^2}
    $$

3.  **Second-order spatial derivative (y-direction)**:
    $$
    \frac{\partial^2 u}{\partial y^2} \approx \frac{u_{i, j+1}^{k} - 2 u_{i, j}^{k} + u_{i, j-1}^{k}}{(\Delta y)^2}
    $$

$$
\frac{u_{i, j}^{k+1} - 2 u_{i, j}^{k} + u_{i, j}^{k-1}}{(\Delta t)^2} = c_{i, j}^2 \left[ \frac{u_{i+1, j}^{k} - 2 u_{i, j}^{k} + u_{i-1, j}^{k}}{h^2} + \frac{u_{i, j+1}^{k} - 2 u_{i, j}^{k} + u_{i, j-1}^{k}}{h^2} \right]
$$
