# Wave Simulation Tutorial

> You can get this simulator code from
> https://github.com/caoqiming/wave-simulator

## Code Structure

- `boundary_conditions.py` implements three simple boundary conditions
  `BoundaryCondition` abstract class defines the interface for boundary conditions. Each boundary condition needs to implement the apply function for both 1D and 2D simulations. Due to symmetry, we only need to implement one directional boundary. Here we implement the left boundary.
  Specific implementations include FixedBoundary, NeumannBoundary, UnlimitedBoundary
- `boundary.py` contains boundary-related logic
  `Boundary` class represents a boundary, including the logic for applying boundaries using symmetry
  `Boundaries` class represents a set of boundaries, including multiple edges and the internal area.
- `source.py` defines wave sources
  `LineSource` class implements a line-segment shaped source
- `oned_simulator.py` one-dimensional simulation
- `twod_simulator.py` two-dimensional simulation

## Double-slit experiment

Let:

- Wavelength: λ
- Distance between slit centers (slit separation): d
- Screen distance (from slits to observation screen): L
- Vertical coordinate from center on screen: y
- Observation angle: θ (angle between slit plane normal and observation direction)

![image](./Optical%20path%20difference.jpeg)

Derivation of optical path difference $\Delta$

Optical path difference at point P: $\Delta = r_2 - r_1$

$$(1) \quad y_1 = y - \frac{d}{2}$$

$$(2) \quad y_2 = y + \frac{d}{2}$$

$$(3) \quad r_1^2 = L^2 + y_1^2$$

$$(4) \quad r_2^2 = L^2 + y_2^2$$

The key point: when $L \gg d, y$

$$(5) \quad r_2 + r_1 \approx 2L$$

From equations (4)-(3):
$$r_2^2 - r_1^2 = y_2^2 - y_1^2$$

Using the difference of squares formula and substituting the approximation from equation (5):
$$(r_2 + r_1)(r_2 - r_1) = 2L \Delta$$

On the other hand, substituting equations (1) and (2):
$$y_2^2 - y_1^2 = (y_2 - y_1)(y_2 + y_1)$$
$$y_2 - y_1 = \left(y + \frac{d}{2}\right) - \left(y - \frac{d}{2}\right) = d$$
$$y_2 + y_1 = \left(y + \frac{d}{2}\right) + \left(y - \frac{d}{2}\right) = 2y$$
Therefore:
$$r_2^2 - r_1^2 = (d)(2y) = 2yd$$

Combining both results:
$$2L \Delta = 2yd$$

$$\Delta = \frac{yd}{L}$$

Geometry and interference conditions:

- Bright fringes (constructive interference) condition: $\Delta = m \lambda, m \in \mathbb{Z}$
- Far-field approximation (Fraunhofer diffraction) and small angle: $\sin\theta \approx \tan\theta \approx y/L$

Thus the vertical position of bright fringes

$$
y_m \approx L \cdot \sin\theta \approx L \cdot (m \lambda / d) = m \cdot (L \lambda / d)
$$

Distance between adjacent bright fringes (difference between m and m+1)

$$
\Delta y = y_{m+1} - y_m \approx L \lambda / d
$$
