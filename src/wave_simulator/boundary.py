from typing import Literal, Tuple, List
from wave_simulator.boundary_conditions import *
import numpy as np
from numpy.typing import NDArray


class Boundary:
    def __init__(
            self,
            type: Literal['up', 'down', 'left', 'right'],
            boundaryCondition: BoundaryCondition,
            start: Tuple[int, int],
            end: Tuple[int, int],
    ):
        """
        boundary 的 上下左右是相对于仿真的区域而言的，比如 left 表示这个边界在仿真区域的左边。
        """
        allowed = {'up', 'down', 'left', 'right'}
        if type not in allowed:
            raise ValueError(
                f"Invalid boundary type '{type}'. Must be one of {allowed}.")
        self.type = type
        if not isinstance(boundaryCondition, BoundaryCondition):
            raise TypeError(
                "boundaryCondition must be an instance of BoundaryCondition")

        def _validate_point(p, name):
            if (not isinstance(p, tuple) or len(p) != 2 or
                    not all(isinstance(c, int) for c in p)):
                raise TypeError(
                    f"{name} must be a tuple of two numbers (x, y)")

        _validate_point(start, 'start')
        _validate_point(end, 'end')

        x1, y1 = start
        x2, y2 = end
        if type in ('up', 'down'):
            if y1 != y2:
                raise ValueError(
                    "For 'up'/'down' boundaries start and end must share y (horizontal).")
            if x1 >= x2:
                raise ValueError(
                    "For 'up'/'down' boundaries start.x must be < end.x (start left of end).")
        elif type in ('left', 'right'):
            if x1 != x2:
                raise ValueError(
                    "For 'left'/'right' boundaries start and end must share x (vertical).")
            if y1 >= y2:
                raise ValueError(
                    "For 'left'/'right' boundaries start.x must be < end.x (start down of end).")

        self.start = start
        self.end = end
        self.boundaryCondition = boundaryCondition

    def apply(
        self,
        u_last: NDArray[np.float64],
        u_current: NDArray[np.float64],
        u_next: NDArray[np.float64],
        C: NDArray[np.float64],
        C2: NDArray[np.float64],
    ) -> None:
        x1, y1 = self.start
        x2, y2 = self.end
        if self.type == "left":
            # left
            u_next[x1, y1:y2+1] = self.boundaryCondition.apply2D(
                u_current[x1, y1:y2+1],
                u_current[x1+1, y1:y2+1],  # 因为是左边界，所以上一组相邻的点向右取得
                C=C[x1, y1:y2+1],
                C2=C2[x1, y1:y2+1],
                u_0_j_last=u_last[x1, y1:y2+1],
            )
        elif self.type == "right":
            # right
            u_next[x1, y1:y2+1] = self.boundaryCondition.apply2D(
                u_current[x1, y1:y2+1],
                u_current[x1-1, y1:y2+1],  # 因为是右边界，所以上一组相邻的点向左取得
                C=C[x1, y1:y2+1],
                C2=C2[x1, y1:y2+1],
                u_0_j_last=u_last[x1, y1:y2+1],
            )
        elif self.type == "up":
            # up
            u_next[x1:x2+1, y1] = self.boundaryCondition.apply2D(
                u_current[x1:x2+1, y1],
                u_current[x1:x2+1, y1-1],  # 因为是上边界，所以上一组相邻的点向下取得
                C=C[x1:x2+1, y1],
                C2=C2[x1:x2+1, y1],
                u_0_j_last=u_last[x1:x2+1, y1],
            )
        elif self.type == "down":
            # down
            u_next[x1:x2+1, y1] = self.boundaryCondition.apply2D(
                u_current[x1:x2+1, y1],
                u_current[x1:x2+1, y1+1],  # 因为是下边界，所以上一组相邻的点向上取得
                C=C[x1:x2+1, y1],
                C2=C2[x1:x2+1, y1],
                u_0_j_last=u_last[x1:x2+1, y1],
            )


class Area:
    def __init__(
        self,
        start: Tuple[int, int],  # 左下角 (x_left, y_bottom)
        end: Tuple[int, int],    # 右上角 (x_right, y_top)
    ):
        x1, y1 = start
        x2, y2 = end
        if not (x1 < x2 and y1 < y2):
            raise ValueError(
                f"internalStart 必须是左下角，internalEnd 必须是右上角 (x1<x2 且 y1<y2),got {start} {end}")
        self.start = start
        self.end = end


class Boundaries:
    def __init__(
        self,
        boundaryList: List[Boundary],
        internalAreas: List[Area] = [],

    ):
        self.boundaries = boundaryList
        self.internalAreas = internalAreas

    def apply(
        self,
        u_last: NDArray[np.float64],
        u_current: NDArray[np.float64],
        u_next: NDArray[np.float64],
        C: NDArray[np.float64],
        C2: NDArray[np.float64],
    ) -> None:
        for b in self.boundaries:
            b.apply(
                u_last,
                u_current,
                u_next,
                C,
                C2
            )

        # mask the internal areas
        for area in self.internalAreas:
            (x1, y1) = area.start
            (x2, y2) = area.end
            u_next[x1:x2+1, y1:y2+1] = 0.0

    def __add__(self, other: 'Boundaries') -> 'Boundaries':
        return Boundaries(
            boundaryList=self.boundaries + other.boundaries,
            internalAreas=self.internalAreas + other.internalAreas
        )


def getDefaultBoundaries(shape: Tuple[int, int]) -> Boundaries:
    w, h = shape
    return Boundaries(
        [
            Boundary("up", UnlimitedBoundary(), (0, h-1), (w-1, h-1)),
            Boundary("down", UnlimitedBoundary(), (0, 0), (w-1, 0)),
            Boundary("left", UnlimitedBoundary(), (0, 0), (0, h-1)),
            Boundary("right", UnlimitedBoundary(), (w-1, 0), (w-1, h-1))
        ]
    )
