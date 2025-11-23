import abc
from typing import Callable, List
import numpy as np
from numpy.typing import NDArray


class Source(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def apply(self, t: float, u_next: NDArray[np.float64]) -> None:
        raise RuntimeError("not implemented")


class Sources():
    def __init__(self, sourceList: List[Source]):
        self.sourceList = sourceList

    def apply(self, t: float, u_next: NDArray[np.float64]) -> None:
        for s in self.sourceList:
            s.apply(t, u_next)

    def __add__(self, other: 'Sources') -> 'Sources':
        return Sources(
            self.sourceList + other.sourceList
        )


class LineSource(Source):
    def __init__(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        f: Callable[[np.float64], np.float64],
    ):
        super().__init__()
        x1, y1 = start
        x2, y2 = end
        if x1 != x2 and y1 != y2:
            raise ValueError("The line should be vertical or horizontal")
        if not (x1 <= x2 and y1 <= y2):
            raise ValueError(
                f"we need x1<=x2 and y1<=y2 ,got {start} {end}")
        self.start = start
        self.end = end
        self.f = f

    def apply(self, t: float, u_next: NDArray[np.float64]) -> None:
        x1, y1 = self.start
        x2, y2 = self.end
        v = self.f(t)
        if x1 == x2:
            u_next[x1, y1:y2+1] = v
        elif y1 == y2:
            u_next[x1:x2+1, y1] = v
        else:
            raise ValueError("The line should be vertical or horizontal")


def getDefaultSources() -> Sources:
    """
    returns the empty sources.
    """
    return Sources([])
