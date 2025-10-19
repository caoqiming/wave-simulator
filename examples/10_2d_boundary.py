from wave_simulator import TwoDimensionSimulator, FixedBoundary, NeumannBoundary, UnlimitedBoundary
import numpy as np

s = TwoDimensionSimulator()


def my_initial_wave(x, y):
    return 0.2*np.exp(-((x-1)**2/0.1 + (y-1)**2/0.1))


s.set_initial_wave(my_initial_wave)
s.set_left_boundary(FixedBoundary())
s.set_right_boundary(FixedBoundary())
s.set_up_boundary(UnlimitedBoundary())
s.set_down_boundary(NeumannBoundary())
s.simulate()
s.animate_result_3D()
