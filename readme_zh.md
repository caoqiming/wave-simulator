# 波形仿真

该项目用于波的仿真。

> 请勿将该项目用于学习以外的任何目的，因为作者不保证其严谨性

## 安装

```bash
pip install wave-simulator
```

## 例子

以下一个简单的例子，你可以在[这里](./examples/)找到更多的例子。

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

## 详细指南

你可以设置仿真空间范围和时间范围，这个设置应该放在其他设置的前面。

```python
s.set_simulation_range(6, 0.01, 16, 0.001)
```

设置初始波形

```python
def my_initial_wave(x):
    if x < 4:
        return 0.5*x
    return -x + 6

s.set_initial_wave(my_initial_wave)
```

设置初始质点的速度

```python
def my_initial_speed(x):
    if x > 0.1 and x < np.pi/4+0.1:
        return -8*np.cos(8*(x-0.1))
    return 0

s.set_initial_point_speed(my_initial_speed)
```

设置波速

```python
def my_wave_speed(x):
    if x > 1.5:
        return 0.5
    return 1

s.set_wave_speed(my_wave_speed)
```

设置边界条件

```python
s.set_left_boundary(FixedBoundary())
s.set_right_boundary(FixedBoundary())
s.set_up_boundary(UnlimitedBoundary())
s.set_down_boundary(NeumannBoundary())
```

将仿真结果保存为视频，只需要添加一个`save_path`参数

```python
s.animate_result_3D(save_path="your_path.mp4")
```

## 数学符号与代码变量的对应关系

| 变量名称     | 含义                  |
| ------------ | --------------------- |
| `L_x`        | 仿真 x 的范围         |
| `dx`         | 仿真 x 的间隔         |
| `X`          | x 的离散的取值        |
| `c`          | 波速                  |
| `c2`         | $c^2$                 |
| `C`          | $c\frac{dt}{dx}$      |
| `C2`         | $(c\frac{dt}{dx})^2 $ |
| `c2_i_sub_1` | $c_{i-1}^2$           |
| `c2_i`       | $c_i^2$               |
| `c2_i_add_1` | $c_{i+1}^2$           |
| `u_i_sub_1`  | $u_{i-1}$             |
| `u_i`        | $u_{i}$               |
| `u_i_add_1`  | $u_{i+1}$             |
| `u_i_j`      | $u_{i,j}$             |
| `u_ip1_j`    | $u_{i+1,j}$           |
| `u_is1_j`    | $u_{i-1,j}$           |
| `u_i_ja1`    | $u_{i,j+1}$           |
| `u_i_js1`    | $u_{i,j-1}$           |

## 相关公式

[1d 参考](./readme_1d_zh.md)
[2d 参考](./readme_2d_zh.md)
