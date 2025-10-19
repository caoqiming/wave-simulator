import numpy as np
from wave_simulator import OneDimensionSimulator


def my_initial_wave(x):
    if x > 0.1 and x < np.pi/4+0.1:
        return np.sin(8*(x-0.1))
    return 0


def my_initial_speed(x):
    if x > 0.1 and x < np.pi/4+0.1:
        return -8*np.cos(8*(x-0.1))
    return 0


def my_wave_speed(x):
    if x > 1.5:
        return 0.5
    return 1


s = OneDimensionSimulator()
s.set_space_range(np.pi, 0.001)
s.set_initial_wave(my_initial_wave)
s.set_initial_point_speed(my_initial_speed)
s.set_wave_speed(my_wave_speed)
s.simulate()
s.animate_result_1D(ylim=(-4, 4), down_sampling_rate=20)
