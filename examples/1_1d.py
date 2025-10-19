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
