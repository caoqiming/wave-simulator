import abc
import numpy as np
from typing import Any


class BoundaryCondition(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def apply(self, u_0: np.float64, u_1: np.float64, **kwargs: Any) -> np.float64:
        """
        Taking the left boundary as an example, input u_0 and u_1 from the previous time step, output u_0 for the next time step.
        """
        raise RuntimeError("not implemented")

    @abc.abstractmethod
    def apply2D(self, u_0_j: np.typing.NDArray, u_1_j: np.typing.NDArray, **kwargs: Any) -> np.typing.NDArray:
        """
        Taking the left boundary as an example, input u_0_j and u_1_j from the previous time step, output u_0_j for the next time step.
        """
        raise RuntimeError("not implemented")


class FixedBoundary(BoundaryCondition):
    """
    Fixed Boundary
    """

    def __init__(self, value=np.float64(0)):
        super().__init__()
        self.value = value

    def apply(self, u_0: np.float64, u_1: np.float64, **kwargs: Any) -> np.float64:
        return self.value

    def apply2D(self, u_0_j: np.typing.NDArray, u_1_j: np.typing.NDArray, **kwargs: Any) -> np.typing.NDArray:
        return np.full(u_0_j.shape, self.value, dtype=np.float64)


class NeumannBoundary(BoundaryCondition):
    """
    The displacement gradient (slope) at the boundary point is zero (e.g., the end of a string is tied to a ring that can slide freely on a rod).
    """

    def apply(self, u_0: np.float64, u_1: np.float64, **kwargs: any) -> np.float64:
        # C=c*dt/dx
        C = kwargs.get("C")
        if C is None:
            raise ValueError("C is not set")
        u_0_last = kwargs.get("u_0_last")
        if u_0_last is None:
            raise ValueError("u_0_last is not set")

        return 2*u_0 - u_0_last + C**2*(u_1-u_0)

    def apply2D(self, u_0_j: np.typing.NDArray, u_1_j: np.typing.NDArray, **kwargs: Any) -> np.typing.NDArray:
        C2 = kwargs.get("C2")
        if C2 is None:
            raise ValueError("C2 is not set")
        u_0_j_last = kwargs.get("u_0_j_last")
        if u_0_j_last is None:
            raise ValueError("u_0_j_last is not set")
        N = u_0_j.shape[0] - 1
        u_0_ja1 = u_0_j[2:N+1]
        u_0_js1 = u_0_j[0:N-1]
        u_0_j_next = np.zeros(u_0_j.shape, dtype=np.float64)
        # Process the intermediate points first
        u_0_j_next[1:N] = 2*u_0_j[1:N] - u_0_j_last[1:N] + C2[1:N] * \
            (u_1_j[1:N]*2 + u_0_ja1 + u_0_js1 - 4*u_0_j[1:N])
        # Unsure how to handle the corner points. Corner points belong to two edges, which is a headache.
        # For now, let's assume the other edge it belongs to is also a NeumannBoundary; the latter one will have higher priority during actual implementation.
        u_0_j_next[0] = 2*u_0_j[0] - u_0_j_last[0] + C2[0] * \
            (u_1_j[0]*2 + 2*u_0_j[1] - 4*u_0_j[0])
        u_0_j_next[N] = 2*u_0_j[N] - u_0_j_last[N] + C2[N] * \
            (u_1_j[N]*2 + 2*u_0_j[N-1] - 4*u_0_j[N])
        return u_0_j_next


class UnlimitedBoundary(BoundaryCondition):
    """
    Unlimited Boundary
    """

    def apply(self, u_0: np.float64, u_1: np.float64, **kwargs: any) -> np.float64:
        C = kwargs.get("C")
        if C is None:
            raise ValueError("C is not set")

        return (1-C)*u_0+C*u_1

    def apply2D(self, u_0_j: np.typing.NDArray, u_1_j: np.typing.NDArray, **kwargs: Any) -> np.typing.NDArray:
        C = kwargs.get("C")
        if C is None:
            raise ValueError("C is not set")

        return (1-C)*u_0_j+C*u_1_j
