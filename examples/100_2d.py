from wave_simulator import TwoDimensionSimulator
import numpy as np

s = TwoDimensionSimulator()


def my_initial_wave(x, y):
    return 0.2*np.exp(-((x-1)**2/0.1 + (y-1)**2/0.1))


s.set_initial_wave(my_initial_wave)
s.simulate()
s.animate_result_3D()
# you can also show result in flat
# s.animate_result_flat()
