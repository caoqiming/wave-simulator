from wave_simulator import TwoDimensionSimulator
from wave_simulator.boundary_conditions import UnlimitedBoundary
import numpy as np

s = TwoDimensionSimulator()


def my_source(t: float) -> float:
    return 5*np.cos(20*t)


s.set_simulation_range(5, 5, 0.02, 8, 0.001)
s.addRectangleBoundaries((2.45, 0), (2.55, 2.4),
                         boundaryCondition=UnlimitedBoundary())
s.addRectangleBoundaries((2.45, 2.6), (2.55, 5),
                         boundaryCondition=UnlimitedBoundary())
s.addLineSource((0, 0), (0, 5), my_source)
s.simulate()
s.animate_result_flat(gamma=0.45, vmin=5, vmax=-5, downsample_temporal=5)
# s.animate_result_flat(gamma=0.45, vmin=5, vmax=-5,
#                       save_path="/Users/glimmer/Documents/study/mathematics_of_light_and_sound/wave-simulation/output/diffraction.mp4")
