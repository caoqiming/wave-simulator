import numpy as np
from typing import Callable, Tuple
from wave_simulator.boundary_conditions import *
from wave_simulator.boundary import Boundaries, Boundary, Area, getDefaultBoundaries
from wave_simulator.source import Sources, LineSource, getDefaultSources
from wave_simulator.animation_utils import animate_result_flat, animate_result_3D


class TwoDimensionSimulator:
    def __init__(self):
        # Space
        self.L_x = 5  # Simulation distance range, from 0 to L_x
        self.dx = 0.05  # Minimum distance interval for simulation
        # Number of segments for the simulation in x
        self.N_x = int(self.L_x/self.dx)
        self.X = np.linspace(0, self.L_x, self.N_x+1,
                             dtype=np.float64)  # Simulation spatial range in x

        self.L_y = 5
        self.dy = self.dx  # Enforce dy=dx
        self.N_y = int(self.L_y/self.dy)
        self.Y = np.linspace(0, self.L_y, self.N_y+1,
                             dtype=np.float64)

        # Time
        self.L_t = 8  # Duration of simulation [s]
        self.dt = 0.005  # Time interval
        self.N_t = int(self.L_t/self.dt)  # Number of time steps
        self.T = np.linspace(0, self.L_t, self.N_t+1)  # Time range

        # Boundary conditions
        self.boundaries = getDefaultBoundaries((self.N_x, self.N_y))
        self.sources = getDefaultSources()

        # Initial waveform, default is all 0
        self.initial_wave = lambda x, y: 0.0
        # Initial speed of each particle, default is 0
        self.initial_point_speed = lambda x, y: 0.0
        # Medium wave speed, default is 1
        self.wave_speed = lambda x, y: 1.0

    def set_simulation_range(self, L_x, L_y, dx, L_t, dt):
        """
        Set simulation range. Notice that boundaries will be reset after this.
        """
        self.L_x = L_x
        self.dx = dx
        self.N_x = int(self.L_x/self.dx)
        self.X = np.linspace(0, self.L_x, self.N_x+1,
                             dtype=np.float64)
        self.L_y = L_y
        self.dy = self.dx
        self.N_y = int(self.L_y/self.dy)
        self.Y = np.linspace(0, self.L_y, self.N_y+1,
                             dtype=np.float64)
        self.L_t = L_t
        self.dt = dt
        self.N_t = int(self.L_t/self.dt)
        self.T = np.linspace(0, self.L_t, self.N_t+1)

        # reset boundaries because map size has changed
        self.boundaries = getDefaultBoundaries((self.N_x, self.N_y))

    def set_initial_wave(
        self,
        initial_wave: Callable[[np.float64, np.float64], np.float64],
    ):
        """
        Sets the initial waveform. Represented by a function that takes the spatial x-coordinate and y-coordinate as input and returns the wave function value at that position.
        """
        self.initial_wave = initial_wave

    def set_initial_point_speed(
        self,
        initial_point_speed: Callable[[np.float64, np.float64], np.float64],
    ):
        """
        Sets the initial speed of each particle. Represented by a function that takes the spatial x-coordinate and y-coordinate as input and returns the initial speed of the particle at that position.
        """
        self.initial_point_speed = initial_point_speed

    def set_wave_speed(
        self,
        wave_speed: Callable[[np.float64, np.float64], np.float64],
    ):
        """
        Sets the wave speed in the medium. Represented by a function that takes the spatial x-coordinate and y-coordinate as input and returns the wave speed in the medium at that position.
        """
        self.wave_speed = wave_speed

    def addRectangleBoundaries(
            self,
            start: Tuple[float, float],
            end: Tuple[float, float],
            boundaryCondition: BoundaryCondition = NeumannBoundary(),
    ) -> None:
        # 将矩形范围的表示从距离转化为索引
        x1, y1 = start
        x2, y2 = end
        x1 = int(x1/self.dx)
        x2 = int(x2/self.dx)
        y1 = int(y1/self.dy)
        y2 = int(y2/self.dy)

        if not (x1 < x2 and y1 < y2):
            raise ValueError(
                "internalStart 必须是左下角，internalEnd 必须是右上角 (x1<x2 且 y1<y2)")

        # 限制到地图范围以内
        if x1 < 0:
            x1 = 0
        if x2 > self.N_x-1:
            x2 = self.N_x-1
        if y1 < 0:
            y1 = 0
        if y2 > self.N_y-1:
            y2 = self.N_y-1

        boundaryList = []
        if x1 > 0:
            boundaryList.append(
                Boundary("right", boundaryCondition, (x1, y1), (x1, y2)))
        if y1 > 0:
            boundaryList.append(
                Boundary("up", boundaryCondition, (x1, y1), (x2, y1)))
        if y2 < self.N_y-1:
            boundaryList.append(
                Boundary("down", boundaryCondition, (x1, y2), (x2, y2)))
        if x2 < self.N_x-1:
            boundaryList.append(
                Boundary("left", boundaryCondition, (x2, y1), (x2, y2)))

        self.boundaries += Boundaries(boundaryList, [Area((x1, y1), (x2, y2))])

    def addLineSource(
            self,
            start: Tuple[float, float],
            end: Tuple[float, float],
            f: Callable[[float], float],
    ):
        x1, y1 = start
        x2, y2 = end
        x1 = int(x1/self.dx)
        x2 = int(x2/self.dx)
        y1 = int(y1/self.dy)
        y2 = int(y2/self.dy)
        # 限制到地图范围以内
        if x1 < 0:
            x1 = 0
        if x2 > self.N_x-1:
            x2 = self.N_x-1
        if y1 < 0:
            y1 = 0
        if y2 > self.N_y-1:
            y2 = self.N_y-1

        self.sources += Sources([LineSource((x1, y1), (x2, y2), f)])

    def simulate(self):
        """
        Starts the simulation
        """
        # Used to store the result
        self.result = np.zeros(
            (self.N_x+1, self.N_y+1, self.N_t+1), np.float64)

        # Used to store the waveform at the current time steps i-1, i, i+1
        u_last = np.zeros((self.N_x+1, self.N_y+1), np.float64)
        u_current = np.zeros((self.N_x+1, self.N_y+1), np.float64)
        u_next = np.zeros((self.N_x+1, self.N_y+1), np.float64)

        # Initialization
        c = np.zeros((self.N_x+1, self.N_y+1), np.float64)
        initial_v = np.zeros((self.N_x+1, self.N_y+1), np.float64)
        for i in range(0, self.N_x+1):
            for j in range(0, self.N_y+1):
                # Medium wave speed
                c[i, j] = self.wave_speed(self.X[i], self.Y[j])
                # Initial waveform
                u_last[i, j] = self.initial_wave(self.X[i], self.Y[j])
                # Initial particle speed
                initial_v[i, j] = self.initial_point_speed(
                    self.X[i], self.Y[j])

        # Define some constants to avoid repeated calculations
        c2 = c**2
        C = c*self.dt/self.dx
        C2 = C**2

        # check CFL
        if np.any(C > 0.1):
            raise ValueError(
                "CFL check failed, you should reduce dt, wave speed or increase dx")

        # For 2D simulation, we use i to represent the x spatial index and j for the y spatial index.
        # The position i,j is [1:self.N_x,1:self.N_y]
        # i+1,j [2:self.N_x+1,1:self.N_y]
        # i-1,j [0:self.N_x-1,1:self.N_y]
        # i,j+1 [1:self.N_x,2:self.N_y+1]
        # i,j-1 [1:self.N_x,0:self.N_y-1]

        u_i_j = u_last[1:self.N_x, 1:self.N_y]
        u_ip1_j = u_last[2:self.N_x+1, 1:self.N_y]
        u_is1_j = u_last[0:self.N_x-1, 1:self.N_y]
        u_i_ja1 = u_last[1:self.N_x, 2:self.N_y+1]
        u_i_js1 = u_last[1:self.N_x, 0:self.N_y-1]
        c2_i_j = c2[1:self.N_x, 1:self.N_y]

        # Calculate initial acceleration
        initial_a = np.zeros((self.N_x+1, self.N_y+1), np.float64)
        initial_a[1:self.N_x, 1:self.N_y] = c2_i_j/self.dx**2 * (
            u_ip1_j + u_is1_j + u_i_ja1 + u_i_js1 - 4*u_i_j
        )
        # Calculate the waveform at t=1
        u_current[1:self.N_x, 1:self.N_y] = u_i_j + \
            initial_v[1:self.N_x, 1:self.N_y] * self.dt +\
            0.5*initial_a[1:self.N_x, 1:self.N_y]*self.dt**2
        # Apply boundary conditions
        self.boundaries.apply(u_last - initial_v * self.dt,
                              u_last, u_current, C, C2)
        # Apply sources
        self.sources.apply(0.0, u_current)

        self.result[:, :, 0] = u_last.copy()
        self.result[:, :, 1] = u_current.copy()
        for i in range(1, self.N_t):
            # Calculate non-boundary points 1,...,N-1
            # i,j   [1:self.N_x,1:self.N_y]
            # i+1,j [2:self.N_x+1,1:self.N_y]
            # i-1,j [0:self.N_x-1,1:self.N_y]
            # i,j+1 [1:self.N_x,2:self.N_y+1]
            # i,j-1 [1:self.N_x,0:self.N_y-1]

            u_last_i_j = u_last[1:self.N_x, 1:self.N_y]
            u_i_j = u_current[1:self.N_x, 1:self.N_y]
            u_ip1_j = u_current[2:self.N_x+1, 1:self.N_y]
            u_is1_j = u_current[0:self.N_x-1, 1:self.N_y]
            u_i_ja1 = u_current[1:self.N_x, 2:self.N_y+1]
            u_i_js1 = u_current[1:self.N_x, 0:self.N_y-1]
            C2_i_j = C2[1:self.N_x, 1:self.N_y]
            u_next_i_j = 2*u_i_j - u_last_i_j + C2_i_j*(
                u_ip1_j + u_is1_j + u_i_ja1 + u_i_js1 - 4*u_i_j
            )

            u_next[1:self.N_x, 1:self.N_y] = u_next_i_j
            # Calculate boundary points
            self.boundaries.apply(u_last, u_current, u_next, C, C2)
            # Apply sources
            self.sources.apply(i*self.dt, u_next)

            self.result[:, :, i+1] = u_next.copy()
            u_last[:] = u_current.copy()
            u_current[:] = u_next.copy()

    def animate_result_flat(self, **args):
        animate_result_flat(self.result, X=self.X, Y=self.Y,
                            boundaries=self.boundaries, **args)

    def animate_result_3D(self, **args):
        animate_result_3D(self.result, X=self.X, Y=self.Y, **args)
