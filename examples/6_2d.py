from wave_simulator import TwoDimensionSimulator
from wave_simulator.boundary_conditions import UnlimitedBoundary
import numpy as np

s = TwoDimensionSimulator()


def my_initial_wave(x, y):
    return 5*np.exp(-((x-1)**2/0.1 + (y-2.5)**2/0.1))


s.set_initial_wave(my_initial_wave)
s.addRectangleBoundaries((2.45, 0), (2.55, 2.4),
                         boundaryCondition=UnlimitedBoundary())
s.addRectangleBoundaries((2.45, 2.6), (2.55, 5),
                         boundaryCondition=UnlimitedBoundary())
s.simulate()
s.animate_result_flat()
