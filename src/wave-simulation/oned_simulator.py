import numpy as np
import abc
from typing import Callable, Any


class BoundaryCondition(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def apply(self, A_0: np.float64, A_1: np.float64, **kwargs: Any) -> np.float64:
        """
        以左边界为例，输入上一时刻的 A_0, A_1，输出下一时刻的 A_0
        """
        raise RuntimeError("not implemented")


class FixedBoundary(BoundaryCondition):
    """
    固定边界
    """

    def __init__(self, value=np.float64(0.0)):
        super().__init__()
        self.value = value

    def apply(self, A_0: np.float64, A_1: np.float64, **kwargs: Any) -> np.float64:
        return self.value


class NeumannBoundary(BoundaryCondition):
    """
    边界点的位移梯度（斜率）为零（例如，一根绳子的末端系在一个可以在杆上自由滑动的环上）。
    """

    def apply(self, A_0: np.float64, A_1: np.float64, **kwargs: any) -> np.float64:
        # C=c*dt/dx
        C = kwargs.get("C")
        if C is None:
            raise ValueError("C is not set")
        A_0_last = kwargs.get("A_0_last")
        if A_0_last is None:
            raise ValueError("A_0_last is not set")

        return 2*A_0 - A_0_last + C**2*(A_1-A_0)


class UnlimitedBoundary(BoundaryCondition):
    """
    无限制边界
    """

    def apply(self, u_0: np.float64, u_1: np.float64, **kwargs: any) -> np.float64:
        C = kwargs.get("C")
        if C is None:
            raise ValueError("C is not set")

        return (1-C)*u_0+C*u_1


class OneDimensionSimulator:
    def __init__(self):
        # 空间
        self.L_x = np.pi/2  # 仿真的距离范围，从 0 到 L_x
        self.dx = 0.01  # 仿真的最小距离间隔
        self.N = int(self.L_x/self.dx)  # 仿真的线段的个数
        self.X = np.linspace(0, self.L_x, self.N+1,
                             dtype=np.float64)  # 仿真的空间范围
        # 时间
        self.L_t = 4  # Duration of simulation [s]
        self.dt = 0.0001  # 时间间隔
        self.N_t = int(self.L_t/self.dt)  # 时间段的个数
        self.T = np.linspace(0, self.L_t, self.N_t+1)  # 时间范围

        # 边界条件
        self.left_boundary = FixedBoundary()
        self.right_boundary = FixedBoundary()

        # 初始波形，默认全为 0
        self.initial_wave = lambda x: 0.0
        # 每个质点的初始速度，默认为 0
        self.initial_point_speed = lambda x: 0.0
        # 介质波速，默认为 1
        self.wave_speed = lambda x: 1.0

    def set_space_range(self, L_x, dx):
        self.L_x = L_x  # 仿真的距离范围，从 0 到 L_x
        self.dx = dx  # 仿真的最小距离间隔
        self.N = int(self.L_x/self.dx)  # 仿真的线段的个数
        self.X = np.linspace(0, self.L_x, self.N+1,
                             dtype=np.float64)  # 仿真的空间范围

    def set_time_range(self, L_t, dt):
        # 时间
        self.L_t = L_t  # Duration of simulation [s]
        self.dt = dt  # 时间间隔
        self.N_t = int(self.L_t/self.dt)  # 时间段的个数
        self.T = np.linspace(0, self.L_t, self.N_t+1)  # 时间范围

    def set_initial_wave(
        self,
        initial_wave: Callable[[np.float64], np.float64],
    ):
        """
        设置初始波形。用一个函数表示，输入空间的横坐标x，返回对应位置的波函数值
        """
        self.initial_wave = initial_wave

    def set_initial_point_speed(
        self,
        initial_point_speed: Callable[[np.float64], np.float64],
    ):
        """
        设置每个质点的初始速度。用一个函数表示，输入空间的横坐标x，返回对应位置质点的初始速度
        """
        self.initial_point_speed = initial_point_speed

    def set_wave_speed(
        self,
        wave_speed: Callable[[np.float64], np.float64],
    ):
        """
        设置介质中的波速。用一个函数表示，输入空间的横坐标x，返回对应位置介质中的波速。
        """
        self.wave_speed = wave_speed

    def set_left_boundary(self, boundary: BoundaryCondition):
        self.left_boundary = boundary

    def set_right_boundary(self, boundary: BoundaryCondition):
        self.right_boundary = boundary

    def simulate(self):
        """
        开始仿真
        """
        # 用于储存结果
        self.result = np.zeros((self.N+1, self.N_t+1), np.float64)

        # 用于储存当前 i-1,i,i+1 时刻的波形
        A_last = np.zeros(self.N+1, np.float64)
        A_current = np.zeros(self.N+1, np.float64)
        A_next = np.zeros(self.N+1, np.float64)
        # 介质波速
        c = np.zeros(self.N+1, np.float64)
        # 质点初始速度
        initial_v = np.zeros(self.N+1, np.float64)
        for i in range(0, self.N+1):
            c[i] = self.wave_speed(self.X[i])
            A_last[i] = self.initial_wave(self.X[i])
            initial_v[i] = self.initial_point_speed(self.X[i])
        # c2 为 c 的平方
        c2 = c**2

        # 仿真需要用 A_last 和 A_current 递推 A_next，所以我们从 t=1 开始仿真
        # 用 t=0 和 t=1 的初始条件，填充 A_last 和 A_current

        initial_a = np.zeros(self.N+1, np.float64)
        # 为了执行效率这里不直接遍历所有的n，而是直接对numpy的array进行操作。可读性较差但没办法，python就是这么垃圾。
        # 所有的点为 0,...,N 能计算的非边界的点为 1,...,N-1
        # 表示 i-1  位置的list的索引为 0,...,N-2  用切片表示就是 [0:N-1]
        # 那么表示 i 位置的list的索引为 1,...,N-1  用切片表示就是 [1:N]
        # 表示 i+1  位置的list的索引为 2,...,N    用切片表示就是 [2:N+1]
        # 理解不了就想想对于同一个索引 i ，在不同的list里的数据是什么

        # (c_i+1)^2
        c2_i_sub_1 = c2[0:self.N-1]
        c2_i = c2[1:self.N]
        c2_i_add_1 = c2[2:self.N+1]

        A_i_sub_1 = A_last[0:self.N-1]
        A_i = A_last[1:self.N]
        A_i_add_1 = A_last[2:self.N+1]
        # 计算初始加速度
        initial_a = np.zeros(self.N+1, np.float64)
        initial_a[1:self.N] = 1/self.dx**2 * (
            0.5*(c2_i_add_1 + c2_i)*(A_i_add_1-A_i)
            - 0.5*(c2_i+c2_i_sub_1)*(A_i-A_i_sub_1))
        # 计算 t=1 时刻的波形
        A_current[1:self.N] = A_last[1:self.N] + \
            initial_v[1:self.N] * self.dt +  \
            0.5 * initial_a[1:self.N] * self.dt**2
        # 应用边界条件
        A_current[0] = self.left_boundary.apply(
            A_last[0], A_last[1],
            C=c[0]*self.dt/self.dx,
            A_0_last=A_last[0] - initial_v[0] * self.dt,
        )
        A_current[self.N] = self.right_boundary.apply(
            A_last[self.N], A_last[self.N-1],
            C=c[self.N]*self.dt/self.dx,
            A_0_last=A_last[self.N] - initial_v[self.N] * self.dt,
        )

        self.result[:, 0] = A_last.copy()
        self.result[:, 1] = A_current.copy()
        for i in range(1, self.N_t):
            # 计算非边界的点 1,...,N-1
            # 表示 i-1  位置的list的索引为 0,...,N-2  用切片表示就是 [0:N-1]
            # 那么表示 i 位置的list的索引为 1,...,N-1  用切片表示就是 [1:N]
            # 表示 i+1  位置的list的索引为 2,...,N    用切片表示就是 [2:N+1]
            A_current_i_sub_1 = A_current[0:self.N-1]
            A_current_i = A_current[1:self.N]
            A_current_i_add_1 = A_current[2:self.N+1]
            A_last_i = A_last[1:self.N]

            A_next_i = 2*A_current_i - A_last_i + (self.dt/self.dx)**2*(
                0.5*(c2_i_add_1+c2_i)*(A_current_i_add_1-A_current_i)
                - 0.5*(c2_i + c2_i_sub_1)*(A_current_i - A_current_i_sub_1)
            )
            A_next[1:self.N] = A_next_i
            # 计算边界的点
            A_next[0] = self.left_boundary.apply(
                A_current[0], A_current[1],
                C=c[0]*self.dt/self.dx,
                A_0_last=A_last[0],
            )
            A_next[self.N] = self.right_boundary.apply(
                A_current[self.N], A_current[self.N-1],
                C=c[self.N]*self.dt/self.dx,
                A_0_last=A_last[self.N],
            )

            self.result[:, i+1] = A_next.copy()
            A_last[:] = A_current.copy()
            A_current[:] = A_next.copy()


if __name__ == "__main__":
    s = OneDimensionSimulator()
    # s.set_space_range(3, 0.01)
    # s.set_initial_wave(lambda x: np.where(x < 2, x, 6-2*x))
    s.set_initial_wave(lambda x: np.where(x < np.pi/4, 3*np.sin(8*x), 0))
    s.set_initial_point_speed(
        lambda x: np.where(x < np.pi/4, -24*np.cos(8*x), 0))

    s.set_left_boundary(UnlimitedBoundary())
    s.set_right_boundary(UnlimitedBoundary())
    s.simulate()
    print(s.result)
    import matplotlib.pyplot as plt
    import viz_tools  # self-developed module that groups animation functions
    anim1 = viz_tools.anim_1D(s.X, s.result, s.dt, 10, save=False,
                              myxlim=(0, s.L_x))
    plt.show()
