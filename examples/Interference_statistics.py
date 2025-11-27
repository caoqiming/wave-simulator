from wave_simulator import TwoDimensionSimulator
from wave_simulator.boundary_conditions import UnlimitedBoundary
from typing import Callable
import numpy as np


def my_source(t: float) -> float:
    return 5*np.cos(20*t)


def init_interfernce_map(
        s: TwoDimensionSimulator,
        distance_between_gaps: float,
        gap_width: float,
        b_w: float = 0.1,
):
    w = s.L_x
    h = s.L_y
    # 障碍物的x坐标
    b_x = 2.5
    s.addRectangleBoundaries((b_x-b_w/2, 0), (b_x+b_w/2, h/2-distance_between_gaps/2-gap_width),
                             boundaryCondition=UnlimitedBoundary())
    s.addRectangleBoundaries((b_x-b_w/2, h/2-distance_between_gaps/2), (b_x+b_w/2, h/2+distance_between_gaps/2),
                             boundaryCondition=UnlimitedBoundary())
    s.addRectangleBoundaries((b_x-b_w/2, h/2+distance_between_gaps/2+gap_width), (b_x+b_w/2, h),
                             boundaryCondition=UnlimitedBoundary())


def test_distance_between_gaps():
    # 测试不同的双缝距离 d
    distance_range = np.arange(1, 3, 0.1)
    for distance in distance_range:
        title = f"distance={distance:.1f}"
        save_path = f"/Users/glimmer/Documents/study/mathematics_of_light_and_sound/wave-simulation/output/distance/distance={distance:.1f}.png"

        s = TwoDimensionSimulator()
        s.set_simulation_range(5, 10, 0.02, 8, 0.001)
        init_interfernce_map(
            s,
            distance_between_gaps=distance,
            gap_width=0.2,
        )

        s.addLineSource((0, 0), (0, 10), my_source)
        s.simulate()
        # s.animate_result_flat(gamma=0.45, vmin=5, vmax=-5, downsample_temporal=5)
        s.show_right_boundary_statistics(
            show=False, title=title, save_path=save_path)


def test_wave_length():
    # 测试不同的波长，仿真的波速默认为1，根据波长设置源的频率即可
    wave_length_range = np.arange(0.1, 1, 0.1)
    for wave_length in wave_length_range:
        title = f"wave_length={wave_length:.1f}"
        save_path = f"/Users/glimmer/Documents/study/mathematics_of_light_and_sound/wave-simulation/output/wave_length/wave_length={wave_length:.1f}.png"

        s = TwoDimensionSimulator()
        s.set_simulation_range(5, 10, 0.02, 8, 0.001)
        init_interfernce_map(
            s,
            distance_between_gaps=1,
            gap_width=0.2,
        )

        def source_with_wave_length(wave_length: float) -> Callable[[float], float]:
            def f(t: float) -> float:
                return 5*np.cos(2*np.pi/wave_length*t)
            return f

        s.addLineSource((0, 0), (0, 10), source_with_wave_length(wave_length))
        s.simulate()
        # s.animate_result_flat(gamma=0.45, vmin=5, vmax=-
        #                       5, downsample_temporal=5)
        s.show_right_boundary_statistics(
            show=False, title=title, save_path=save_path)


def test_L():
    # 测试屏幕到双缝的距离
    L_range = np.arange(1, 5, 0.2)
    for L in L_range:
        title = f"L={L:.1f}"
        save_path = f"/Users/glimmer/Documents/study/mathematics_of_light_and_sound/wave-simulation/output/L/L={L:.1f}.png"

        s = TwoDimensionSimulator()
        s.set_simulation_range(2.5+L, 10, 0.02, 8, 0.001)
        init_interfernce_map(
            s,
            distance_between_gaps=1,
            gap_width=0.2,
        )

        s.addLineSource((0, 0), (0, 10), my_source)
        s.simulate()
        # s.animate_result_flat(gamma=0.45, vmin=5, vmax=-
        #                       5, downsample_temporal=5)
        s.show_right_boundary_statistics(
            show=False, title=title, save_path=save_path)


# test_distance_between_gaps()
# test_wave_length()
test_L()
