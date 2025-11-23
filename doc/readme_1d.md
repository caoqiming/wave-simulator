# 1D Wave Simulation

## Wave Equation in Non-Uniform Media

In a non-uniform medium, the wave speed $c$ is no longer a constant but a function of position, $c(\mathbf{x})$. The $c^2$ term in the standard wave equation cannot be directly moved outside the partial derivative.

Taking the one-dimensional wave equation as an example, when the wave speed $c(x)$ varies with position, the equation is typically written as:

$$
\frac{\partial^2 A}{\partial t^2} = \frac{\partial}{\partial x} \left[ c^2(x) \frac{\partial A}{\partial x} \right]
$$

Finite Difference Form:

$$
\frac{A_{i}^{j+1} - 2A_{i}^{j} + A_{i}^{j-1}}{\Delta t^2} = \frac{1}{\Delta x^2} \left[ \frac{c_{i+1}^2 + c_{i}^2}{2} (A_{i+1}^{j} - A_{i}^{j}) - \frac{c_{i}^2 + c_{i-1}^2}{2} (A_{i}^{j} - A_{i-1}^{j}) \right]
$$

> For the derivation process, [refer here](./wave_equation.md)

### First Derivative Approximation

**Differential Equation Form:**

$$
\frac{\partial}{\partial t} A(x, t) = u
$$

**Finite Difference Form (using $a_j^i$ notation):**

$$
\frac{a_j^{i+1} - a_j^{i-1}}{2\Delta t} \approx u
$$

## Starting Problem

At the start of the simulation, we can specify the waveform for the first frame and the instantaneous velocity of each particle in the first frame. However, the recursive formula derived from the wave equation requires data from two frames to start.
We use the current particle velocity (specified by the initial value) and the acceleration (the right-hand side of the wave equation) to estimate the particle's position in the second frame.

## Boundary Conditions

### Fixed Boundary

The simplest boundary condition, fixed at 0 by default.

### Free End Boundary (Neumann Boundary Condition)

Physical meaning: The displacement gradient (slope) at the boundary point is zero (e.g., the end of a string is tied to a ring that can slide freely on a rod). This simulates a case where the wave reflects without a phase inversion.

We use a **fictitious point** $u_{-1}^n$ to satisfy the Neumann condition:

Neumann condition finite difference approximation:

$$
\frac{\partial u}{\partial x} \Big|_{i=0} \approx \frac{u_1^n - u_{-1}^n}{2 \Delta x} = 0 \quad \Rightarrow \quad u_{-1}^n = u_1^n
$$

Substitute $u_{-1}^n = u_1^n$ into the interior update formula at $i=0$:

### First-order Absorbing Boundary Condition (First-order ABC)

This condition is based on the fact that, in an infinite 1D space with a wave speed of $c$, the wave propagates only in one direction (e.g., at the right boundary, the wave propagates only to the right).

For the left boundary ($x=0$):

A left-propagating wave satisfies $\left(\frac{\partial}{\partial t} - c \frac{\partial}{\partial x}\right) A = 0$.

Therefore, the absorbing boundary condition for the left boundary is:
$$\frac{\partial A}{\partial t} - c \frac{\partial A}{\partial x} = 0 \quad \text{at } x=0$$
Finite difference form:

$$
\frac{A_0^{n+1} - A_0^n}{\Delta t} - c \frac{A_1^n - A_0^n}{\Delta x} = 0
$$

$$
A_0^{n+1} = A_0^n + c \frac{\Delta t}{\Delta x} (A_1^n - A_0^n)
$$

$$
A_0^{n+1} = A_0^n + C (A_1^n - A_0^n)
$$

$$
A_0^{n+1} = (1 - C) A_0^n + C A_1^n
$$
