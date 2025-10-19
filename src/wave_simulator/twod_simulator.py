import numpy as np
from typing import Callable
from wave_simulator.boundary_conditions import *
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
        self.left_boundary = NeumannBoundary()
        self.right_boundary = NeumannBoundary()
        self.up_boundary = NeumannBoundary()
        self.down_boundary = NeumannBoundary()

        # Initial waveform, default is all 0
        self.initial_wave = lambda x, y: 0.0
        # Initial speed of each particle, default is 0
        self.initial_point_speed = lambda x, y: 0.0
        # Medium wave speed, default is 1
        self.wave_speed = lambda x, y: 1.0

    def set_simulation_range(self, L_x, L_y, dx, L_t, dt):
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

    def set_space_range(self, L_x, dx):
        self.L_x = L_x  # Simulation distance range, from 0 to L_x
        self.dx = dx  # Minimum distance interval for simulation
        # Number of segments for the simulation in x
        self.N_x = int(self.L_x/self.dx)
        self.X = np.linspace(0, self.L_x, self.N_x+1,
                             dtype=np.float64)  # Simulation spatial range in x

    def set_time_range(self, L_t, dt):
        # Time
        self.L_t = L_t  # Duration of simulation [s]
        self.dt = dt  # Time interval
        self.N_t = int(self.L_t/self.dt)  # Number of time steps
        self.T = np.linspace(0, self.L_t, self.N_t+1)  # Time range

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

    def set_all_boundary(self, boundary: BoundaryCondition):
        self.left_boundary = boundary
        self.right_boundary = boundary
        self.up_boundary = boundary
        self.down_boundary = boundary

    def set_left_boundary(self, boundary: BoundaryCondition):
        self.left_boundary = boundary

    def set_right_boundary(self, boundary: BoundaryCondition):
        self.right_boundary = boundary

    def set_up_boundary(self, boundary: BoundaryCondition):
        self.up_boundary = boundary

    def set_down_boundary(self, boundary: BoundaryCondition):
        self.down_boundary = boundary

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
        # left i=0
        u_current[0, :] = self.left_boundary.apply2D(
            u_last[0, :], u_last[1, :],
            C=C[0, :],
            C2=C2[0, :],
            u_0_j_last=u_last[0, :] - initial_v[0, :] * self.dt,
        )
        # right i=N
        u_current[self.N_x, :] = self.right_boundary.apply2D(
            u_last[self.N_x, :], u_last[self.N_x-1, :],
            C=C[self.N_x, :],
            C2=C2[self.N_x, :],
            u_0_j_last=u_last[self.N_x, :] - initial_v[self.N_x, :] * self.dt,
        )
        # up j=N
        u_current[:, self.N_y] = self.up_boundary.apply2D(
            u_last[:, self.N_y], u_last[:, self.N_y-1],
            C=C[:, self.N_y],
            C2=C2[:, self.N_y],
            u_0_j_last=u_last[:, self.N_y] - initial_v[:, self.N_y] * self.dt,
        )
        # down j=0
        u_current[:, 0] = self.down_boundary.apply2D(
            u_last[:, 0], u_last[:, 1],
            C=C[:, 0],
            C2=C2[:, 0],
            u_0_j_last=u_last[:, 0] - initial_v[:, 0] * self.dt,
        )

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
            # left i=0
            u_next[0, :] = self.left_boundary.apply2D(
                u_current[0, :], u_current[1, :],
                C=C[0, :],
                C2=C2[0, :],
                u_0_j_last=u_last[0, :],
            )
            # right i=N
            u_next[self.N_x, :] = self.right_boundary.apply2D(
                u_current[self.N_x, :], u_current[self.N_x-1, :],
                C=C[self.N_x, :],
                C2=C2[self.N_x, :],
                u_0_j_last=u_last[self.N_x, :],
            )
            # up j=N
            u_next[:, self.N_y] = self.up_boundary.apply2D(
                u_current[:, self.N_y], u_current[:, self.N_y-1],
                C=C[:, self.N_y],
                C2=C2[:, self.N_y],
                u_0_j_last=u_last[:, self.N_y],
            )
            # down j=0
            u_next[:, 0] = self.down_boundary.apply2D(
                u_current[:, 0], u_current[:, 1],
                C=C[:, 0],
                C2=C2[:, 0],
                u_0_j_last=u_last[:, 0],
            )

            self.result[:, :, i+1] = u_next.copy()
            u_last[:] = u_current.copy()
            u_current[:] = u_next.copy()

    def animate_result_flat(self, **args):
        animate_result_flat(self.result, X=self.X, Y=self.Y, **args)

    def animate_result_3D(self, **args):
        animate_result_3D(self.result, X=self.X, Y=self.Y, **args)
