from wave_simulator import TwoDimensionSimulator, UnlimitedBoundary
import numpy as np

s = TwoDimensionSimulator()


def my_initial_wave(x, y):
    return 0.5*np.exp(-((x-1)**2/0.1 + (y-4)**2/0.1))


def my_wave_speed(x, y):
    if y < 2.5:
        return 0.5
    return 1


s.set_initial_wave(my_initial_wave)
s.set_wave_speed(my_wave_speed)
s.set_all_boundary(UnlimitedBoundary())
s.simulate()
s.animate_result_flat()
