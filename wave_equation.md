### Continuous Form of the Wave Equation in Non-Uniform Media (1D)

The continuous form we use is:
$$\frac{\partial^2 A}{\partial t^2} = \frac{\partial}{\partial x} \left[ c^2(x) \frac{\partial A}{\partial x} \right]$$

Where $A$ is the wave field and $c(x)$ is the wave speed that varies with position.

### Finite Difference Approximation

We adopt standard notation:

- $A_{i}^{j}$: The value of the wave field at spatial point $x_i$ and time $t_j$.
- $c_{i}$: The value of the wave speed at spatial point $x_i$ (varies with position, constant in time).
- $\Delta t$: The time step size.
- $\Delta x$: The spatial step size.

#### 1. Approximation of the Second-order Time Derivative $\frac{\partial^2 A}{\partial t^2}$

Using central difference:
$$\frac{\partial^2 A}{\partial t^2} \approx \frac{A_{i}^{j+1} - 2A_{i}^{j} + A_{i}^{j-1}}{\Delta t^2}$$

#### 2. Approximation of the Spatial Term $\frac{\partial}{\partial x} \left[ c^2(x) \frac{\partial A}{\partial x} \right]$

For the term containing $c^2(x)$, we use a two-step central difference to approximate it:

**Step 1: Approximate the term inside the brackets $\left[ c^2(x) \frac{\partial A}{\partial x} \right]$**

We first need to define the value of **$c^2(x) \frac{\partial A}{\partial x}$ at the half-grid point $x_{i+1/2}$**.

At $x_{i+1/2}$:
$$\left[ c^2 \frac{\partial A}{\partial x} \right]_{i+1/2}^{j} \approx (c^2)_{i+1/2} \frac{A_{i+1}^{j} - A_{i}^{j}}{\Delta x}$$

To calculate $(c^2)_{i+1/2}$, the **average** of the adjacent grid points $c_{i}$ and $c_{i+1}$ is typically used (or simply the average of their squares):
$$(c^2)_{i+1/2} \approx \frac{c_{i+1}^2 + c_{i}^2}{2}$$

Thus, the first-step approximation is:
$$F_{i+1/2}^j = \frac{c_{i+1}^2 + c_{i}^2}{2} \frac{A_{i+1}^{j} - A_{i}^{j}}{\Delta x}$$

**Step 2: Approximate the outer $\frac{\partial}{\partial x}$**

Now, we apply central difference to the result from Step 1 (which can be viewed as a flux $F$):
$$\frac{\partial}{\partial x} \left[ c^2 \frac{\partial A}{\partial x} \right] \approx \frac{F_{i+1/2}^j - F_{i-1/2}^j}{\Delta x}$$

Where $F_{i-1/2}^j$ is:
$$F_{i-1/2}^j = \frac{c_{i}^2 + c_{i-1}^2}{2} \frac{A_{i}^{j} - A_{i-1}^{j}}{\Delta x}$$

Substituting $F_{i+1/2}^j$ and $F_{i-1/2}^j$, we get the final approximation for the spatial term:
$$\frac{\partial}{\partial x} \left[ c^2 \frac{\partial A}{\partial x} \right] \approx \frac{1}{\Delta x^2} \left[ \frac{c_{i+1}^2 + c_{i}^2}{2} (A_{i+1}^{j} - A_{i}^{j}) - \frac{c_{i}^2 + c_{i-1}^2}{2} (A_{i}^{j} - A_{i-1}^{j}) \right]$$

### 3. Complete Finite Difference Equation (FDM Formula)

Substituting the time and spatial approximations into the continuous equation $\frac{\partial^2 A}{\partial t^2} = \frac{\partial}{\partial x} \left[ c^2 \frac{\partial A}{\partial x} \right]$:

$$\frac{A_{i}^{j+1} - 2A_{i}^{j} + A_{i}^{j-1}}{\Delta t^2} = \frac{1}{\Delta x^2} \left[ \frac{c_{i+1}^2 + c_{i}^2}{2} (A_{i+1}^{j} - A_{i}^{j}) - \frac{c_{i}^2 + c_{i-1}^2}{2} (A_{i}^{j} - A_{i-1}^{j}) \right]$$

**Solving for $A_{i}^{j+1}$ (Time-marching form):**

Let $\lambda = \left(\frac{\Delta t}{\Delta x}\right)^2$. Rearranging gives the final explicit Finite Difference Method formula:

$$A_{i}^{j+1} = 2A_{i}^{j} - A_{i}^{j-1} + \frac{\lambda}{2} \left[ (c_{i+1}^2 + c_{i}^2) (A_{i+1}^{j} - A_{i}^{j}) - (c_{i}^2 + c_{i-1}^2) (A_{i}^{j} - A_{i-1}^{j}) \right]$$
