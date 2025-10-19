from wave_simulator import TwoDimensionSimulator, UnlimitedBoundary
import numpy as np

s = TwoDimensionSimulator()


def my_initial_wave(x, y):
    return 0.2*np.exp(-((x-1)**2/0.1 + (y-1)**2/0.1))


s.set_initial_wave(my_initial_wave)
s.set_all_boundary(UnlimitedBoundary())
s.simulate()
s.animate_result_3D()
