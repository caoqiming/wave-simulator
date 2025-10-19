import numpy as np
from wave_simulator import OneDimensionSimulator


def my_initial_wave(x):
    if x > 2 and x < np.pi/4+2:
        return np.sin(8*(x-2))
    return 0


def my_initial_speed(x):
    if x > 2 and x < np.pi/4+2:
        return 8*np.cos(8*(x-2))
    return 0


s = OneDimensionSimulator()
s.set_space_range(np.pi, 0.01)
s.set_initial_wave(my_initial_wave)
s.set_initial_point_speed(my_initial_speed)
s.simulate()
s.animate_result_1D(ylim=(-4, 4), down_sampling_rate=20)
