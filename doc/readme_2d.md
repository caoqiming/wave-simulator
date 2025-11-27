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

## Fourth-Order Spatial Approximation

To gain higher spatial accuracy, the Laplacian can be approximated with fourth-order central differences while keeping a second-order time update (explicit, good balance of cost and stability).

Fourth-order second derivatives:

- x-direction:
  $$
  \frac{\partial^2 u}{\partial x^2}\Big|_{i,j} \approx \frac{-u_{i-2,j} + 16u_{i-1,j} - 30u_{i,j} + 16u_{i+1,j} - u_{i+2,j}}{12\,h^2}
  $$
- y-direction:
  $$
  \frac{\partial^2 u}{\partial y^2}\Big|_{i,j} \approx \frac{-u_{i,j-2} + 16u_{i,j-1} - 30u_{i,j} + 16u_{i,j+1} - u_{i,j+2}}{12\,h^2}
  $$

Insert into the wave equation (uniform grid $\Delta x = \Delta y = h$):

$$
\frac{u_{i,j}^{k+1} - 2u_{i,j}^{k} + u_{i,j}^{k-1}}{(\Delta t)^2}
= c_{i,j}^2 \left[
\frac{-u_{i-2,j}^{k} + 16u_{i-1,j}^{k} - 30u_{i,j}^{k} + 16u_{i+1,j}^{k} - u_{i+2,j}^{k}}{12\,h^2} +
\frac{-u_{i,j-2}^{k} + 16u_{i,j-1}^{k} - 30u_{i,j}^{k} + 16u_{i,j+1}^{k} - u_{i,j+2}^{k}}{12\,h^2}
\right]
$$

Explicit update:

$$
u_{i,j}^{k+1} = 2u_{i,j}^{k} - u_{i,j}^{k-1} +
\left(\frac{c_{i,j}\Delta t}{h}\right)^2
\frac{-u_{i-2,j}^{k} +16u_{i-1,j}^{k} -60u_{i,j}^{k} +16u_{i+1,j}^{k} -u_{i+2,j}^{k}
 -u_{i,j-2}^{k} +16u_{i,j-1}^{k} +16u_{i,j+1}^{k} -u_{i,j+2}^{k}}{12}
$$

(You may leave the two -30u terms uncombined if clearer.)

Notes:

- Boundary / ghost points: stencil uses offsets ±2, so ghost cells or special boundary closures are required (fixed, free, absorbing).
- Stability (constant c): CFL constraint tightens; a conservative choice
  $$
  \frac{c\,\Delta t}{h} \lesssim \frac{1}{\sqrt{2}}
  $$
  Actual limit depends on implementation and boundary treatment; verify numerically.
- Time order: Keeping time second-order avoids more complex multi-step or Runge–Kutta–Nyström schemes. Higher time order is possible but adds cost and analysis burden.
- Smooth initial data: To realize fourth-order spatial accuracy the initial displacement and velocity must be sufficiently smooth (bounded higher derivatives).
- Wider stencil: Increases memory access and boundary handling complexity.
