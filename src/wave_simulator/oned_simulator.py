import numpy as np
from typing import Callable
from wave_simulator.boundary_conditions import *
from wave_simulator.animation_utils import animate_result_1D


class OneDimensionSimulator:
    def __init__(self):
        # 空间
        self.L_x = np.pi/2  # 仿真的距离范围，从 0 到 L_x
        self.dx = 0.001     # 仿真的最小距离间隔
        self.N = int(self.L_x/self.dx)  # 仿真的线段的个数
        self.X = np.linspace(0, self.L_x, self.N+1,
                             dtype=np.float64)  # 仿真的空间范围
        # 时间
        self.L_t = 4  # Duration of simulation [s]
        self.dt = 0.0001  # 时间间隔
        self.N_t = int(self.L_t/self.dt)  # 时间段的个数
        self.T = np.linspace(0, self.L_t, self.N_t+1)  # 时间范围

        # 边界条件
        self.left_boundary = UnlimitedBoundary()
        self.right_boundary = UnlimitedBoundary()

        # 初始波形，默认全为 0
        self.initial_wave = lambda x: 0.0
        # 每个质点的初始速度，默认为 0
        self.initial_point_speed = lambda x: 0.0
        # 介质波速，默认为 1
        self.wave_speed = lambda x: 1.0

    def set_simulation_range(self, L_x, dx, L_t, dt):
        self.L_x = L_x
        self.dx = dx
        self.N = int(self.L_x/self.dx)
        self.X = np.linspace(0, self.L_x, self.N+1,
                             dtype=np.float64)
        self.L_t = L_t
        self.dt = dt
        self.N_t = int(self.L_t/self.dt)
        self.T = np.linspace(0, self.L_t, self.N_t+1)

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

    def set_all_boundary(self, boundary: BoundaryCondition):
        self.left_boundary = boundary
        self.right_boundary = boundary

    def simulate(self):
        """
        开始仿真
        """
        # 用于储存结果
        self.result = np.zeros((self.N+1, self.N_t+1), np.float64)

        # 用于储存当前 i-1,i,i+1 时刻的波形
        u_last = np.zeros(self.N+1, np.float64)
        u_current = np.zeros(self.N+1, np.float64)
        u_next = np.zeros(self.N+1, np.float64)
        # 介质波速
        c = np.zeros(self.N+1, np.float64)
        # 质点初始速度
        initial_v = np.zeros(self.N+1, np.float64)
        for i in range(0, self.N+1):
            c[i] = self.wave_speed(self.X[i])
            u_last[i] = self.initial_wave(self.X[i])
            initial_v[i] = self.initial_point_speed(self.X[i])

        # check CFL
        C = c*self.dt/self.dx
        if np.any(C > 0.1):
            raise ValueError(
                "CFL check failed, you should reduce dt, wave speed or increase dx")

        # 仿真需要用 u_last 和 u_current 递推 u_next，所以我们从 t=1 开始仿真
        # 用 t=0 和 t=1 的初始条件，填充 u_last 和 u_current

        initial_a = np.zeros(self.N+1, np.float64)
        # 为了执行效率这里不直接遍历所有的n，而是直接对numpy的array进行操作。可读性较差但没办法，python就是这么垃圾。
        # 所有的点为 0,...,N 能计算的非边界的点为 1,...,N-1
        # 表示 i-1  位置的list的索引为 0,...,N-2  用切片表示就是 [0:N-1]
        # 那么表示 i 位置的list的索引为 1,...,N-1  用切片表示就是 [1:N]
        # 表示 i+1  位置的list的索引为 2,...,N    用切片表示就是 [2:N+1]
        # 理解不了就想想对于同一个索引 i ，在不同的list里的数据是什么

        c2 = c**2
        c2_i_sub_1 = c2[0:self.N-1]
        c2_i = c2[1:self.N]
        c2_i_add_1 = c2[2:self.N+1]

        u_i_sub_1 = u_last[0:self.N-1]
        u_i = u_last[1:self.N]
        u_i_add_1 = u_last[2:self.N+1]
        # 计算初始加速度
        initial_a = np.zeros(self.N+1, np.float64)
        initial_a[1:self.N] = 1/self.dx**2 * (
            0.5*(c2_i_add_1 + c2_i)*(u_i_add_1-u_i)
            - 0.5*(c2_i+c2_i_sub_1)*(u_i-u_i_sub_1))
        # 计算 t=1 时刻的波形
        u_current[1:self.N] = u_last[1:self.N] + \
            initial_v[1:self.N] * self.dt +  \
            0.5 * initial_a[1:self.N] * self.dt**2
        # 应用边界条件
        u_current[0] = self.left_boundary.apply(
            u_last[0], u_last[1],
            C=c[0]*self.dt/self.dx,
            u_0_last=u_last[0] - initial_v[0] * self.dt,
        )
        u_current[self.N] = self.right_boundary.apply(
            u_last[self.N], u_last[self.N-1],
            C=c[self.N]*self.dt/self.dx,
            u_0_last=u_last[self.N] - initial_v[self.N] * self.dt,
        )

        self.result[:, 0] = u_last.copy()
        self.result[:, 1] = u_current.copy()
        for i in range(1, self.N_t):
            # 计算非边界的点 1,...,N-1
            # 表示 i-1  位置的list的索引为 0,...,N-2  用切片表示就是 [0:N-1]
            # 那么表示 i 位置的list的索引为 1,...,N-1  用切片表示就是 [1:N]
            # 表示 i+1  位置的list的索引为 2,...,N    用切片表示就是 [2:N+1]
            u_current_i_sub_1 = u_current[0:self.N-1]
            u_current_i = u_current[1:self.N]
            u_current_i_add_1 = u_current[2:self.N+1]
            u_last_i = u_last[1:self.N]

            u_next_i = 2*u_current_i - u_last_i + (self.dt/self.dx)**2*(
                0.5*(c2_i_add_1+c2_i)*(u_current_i_add_1-u_current_i)
                - 0.5*(c2_i + c2_i_sub_1)*(u_current_i - u_current_i_sub_1)
            )
            u_next[1:self.N] = u_next_i
            # 计算边界的点
            u_next[0] = self.left_boundary.apply(
                u_current[0], u_current[1],
                C=c[0]*self.dt/self.dx,
                u_0_last=u_last[0],
            )
            u_next[self.N] = self.right_boundary.apply(
                u_current[self.N], u_current[self.N-1],
                C=c[self.N]*self.dt/self.dx,
                u_0_last=u_last[self.N],
            )

            self.result[:, i+1] = u_next.copy()
            u_last[:] = u_current.copy()
            u_current[:] = u_next.copy()

    def animate_result_1D(self, **args):
        animate_result_1D(self.result, self.X, self.dt,
                          xlim=(0, self.L_x), **args)
