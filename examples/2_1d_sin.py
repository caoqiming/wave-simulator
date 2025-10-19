import numpy as np
from wave_simulator import OneDimensionSimulator, FixedBoundary


def my_initial_wave(x):
    return np.sin(8*x)


s = OneDimensionSimulator()
s.set_initial_wave(my_initial_wave)
s.set_all_boundary(FixedBoundary())
s.simulate()
s.animate_result_1D(ylim=(-4, 4), down_sampling_rate=20)
