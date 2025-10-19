# Wave Simulator

[中文](./readme_zh.md)

This project is for wave simulation.

> **Please do not use this project for any purpose other than learning, as the author does not guarantee its rigor.**

## Installation

```bash
pip install wave-simulator
```

## Examples

The following is a simple example. You can find more examples [here](./examples/).

```python
from wave_simulator import OneDimensionSimulator, FixedBoundary

def my_initial_wave(x):
    if x < 4:
        return 0.5*x
    return -x + 6

s = OneDimensionSimulator()
s.set_simulation_range(6, 0.01, 16, 0.001)
s.set_initial_wave(my_initial_wave)
s.set_left_boundary(FixedBoundary())
s.set_right_boundary(FixedBoundary())
s.simulate()
s.animate_result_1D(ylim=(-4, 4), down_sampling_rate=20)
```

## Detailed Guide

You can set the simulation's spatial and temporal ranges. This setting should be done before other configurations.

```python
s.set_simulation_range(6, 0.01, 16, 0.001)
```

Set the initial waveform

```python
def my_initial_wave(x):
    if x < 4:
        return 0.5*x
    return -x + 6

s.set_initial_wave(my_initial_wave)
```

Set the initial speed of the particles

```python
def my_initial_speed(x):
    if x > 0.1 and x < np.pi/4+0.1:
        return -8*np.cos(8*(x-0.1))
    return 0

s.set_initial_point_speed(my_initial_speed)
```

Set the wave speed

```python
def my_wave_speed(x):
    if x > 1.5:
        return 0.5
    return 1

s.set_wave_speed(my_wave_speed)
```

Set boundary conditions

```python
s.set_left_boundary(FixedBoundary())
s.set_right_boundary(FixedBoundary())
s.set_up_boundary(UnlimitedBoundary())
s.set_down_boundary(NeumannBoundary())
```

To save the simulation result as a video, just add a `save_path` parameter

```python
s.animate_result_3D(save_path="your_path.mp4")
```

## Correspondence between Mathematical Symbols and Code Variables

| Variable Name | Meaning                |
| ------------- | ---------------------- |
| `L_x`         | Simulation range for x |
| `dx`          | Spatial step for x     |
| `X`           | Discrete values for x  |
| `c`           | Wave speed             |
| `c2`          | $c^2$                  |
| `C`           | $c\frac{dt}{dx}$       |
| `C2`          | $(c\\frac{dt}{dx})^2 $ |
| `c2_i_sub_1`  | $c_{i-1}^2$            |
| `c2_i`        | $c_i^2$                |
| `c2_i_add_1`  | $c_{i+1}^2$            |
| `u_i_sub_1`   | $u_{i-1}$              |
| `u_i`         | $u_{i}$                |
| `u_i_add_1`   | $u_{i+1}$              |
| `u_i_j`       | $u_{i,j}$              |
| `u_ip1_j`     | $u_{i+1,j}$            |
| `u_is1_j`     | $u_{i-1,j}$            |
| `u_i_ja1`     | $u_{i,j+1}$            |
| `u_i_js1`     | $u_{i,j-1}$            |

## Related Formulas

[1d reference](./readme_1d.md)

[2d reference](./readme_2d.md)
