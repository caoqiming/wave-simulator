import numpy as np
from typing import Callable
from wave_simulator.boundary_conditions import *
from wave_simulator.animation_utils import animate_result_1D


class OneDimensionSimulator:
    def __init__(self):
        # Space
        self.L_x = np.pi/2  # Simulation distance range, from 0 to L_x
        self.dx = 0.001     # Minimum distance interval for simulation
        self.N = int(self.L_x/self.dx)  # Number of segments for the simulation
        self.X = np.linspace(0, self.L_x, self.N+1,
                             dtype=np.float64)  # Simulation spatial range
        # Time
        self.L_t = 4  # Duration of simulation [s]
        self.dt = 0.0001  # Time interval
        self.N_t = int(self.L_t/self.dt)  # Number of time steps
        self.T = np.linspace(0, self.L_t, self.N_t+1)  # Time range

        # Boundary conditions
        self.left_boundary = UnlimitedBoundary()
        self.right_boundary = UnlimitedBoundary()

        # Initial waveform, default is all 0
        self.initial_wave = lambda x: 0.0
        # Initial speed of each particle, default is 0
        self.initial_point_speed = lambda x: 0.0
        # Medium wave speed, default is 1
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
        self.L_x = L_x  # Simulation distance range, from 0 to L_x
        self.dx = dx  # Minimum distance interval for simulation
        self.N = int(self.L_x/self.dx)  # Number of segments for the simulation
        self.X = np.linspace(0, self.L_x, self.N+1,
                             dtype=np.float64)  # Simulation spatial range

    def set_time_range(self, L_t, dt):
        # Time
        self.L_t = L_t  # Duration of simulation [s]
        self.dt = dt  # Time interval
        self.N_t = int(self.L_t/self.dt)  # Number of time steps
        self.T = np.linspace(0, self.L_t, self.N_t+1)  # Time range

    def set_initial_wave(
        self,
        initial_wave: Callable[[np.float64], np.float64],
    ):
        """
        Sets the initial waveform. Represented by a function that takes the spatial x-coordinate as input and returns the wave function value at that position.
        """
        self.initial_wave = initial_wave

    def set_initial_point_speed(
        self,
        initial_point_speed: Callable[[np.float64], np.float64],
    ):
        """
        Sets the initial speed of each particle. Represented by a function that takes the spatial x-coordinate as input and returns the initial speed of the particle at that position.
        """
        self.initial_point_speed = initial_point_speed

    def set_wave_speed(
        self,
        wave_speed: Callable[[np.float64], np.float64],
    ):
        """
        Sets the wave speed in the medium. Represented by a function that takes the spatial x-coordinate as input and returns the wave speed in the medium at that position.
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
        Starts the simulation
        """
        # Used to store the result
        self.result = np.zeros((self.N+1, self.N_t+1), np.float64)

        # Used to store the waveform at the current time steps i-1, i, i+1
        u_last = np.zeros(self.N+1, np.float64)
        u_current = np.zeros(self.N+1, np.float64)
        u_next = np.zeros(self.N+1, np.float64)
        # Medium wave speed
        c = np.zeros(self.N+1, np.float64)
        # Initial particle speed
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

        # The simulation needs u_last and u_current to iterate u_next, so we start the simulation from t=1.
        # Fill u_last and u_current with initial conditions for t=0 and t=1.

        initial_a = np.zeros(self.N+1, np.float64)
        # For execution efficiency, instead of directly iterating through all n, we operate directly on numpy arrays. This reduces readability but is unavoidable in Python.
        # All points are 0,...,N. The non-boundary points that can be calculated are 1,...,N-1.
        # The indices for the list representing position i-1 are 0,...,N-2, which is [0:N-1] using slicing.
        # The indices for the list representing position i are 1,...,N-1, which is [1:N] using slicing.
        # The indices for the list representing position i+1 are 2,...,N, which is [2:N+1] using slicing.
        # If this is confusing, think about what data corresponds to the same index i in different lists.

        c2 = c**2
        c2_i_sub_1 = c2[0:self.N-1]
        c2_i = c2[1:self.N]
        c2_i_add_1 = c2[2:self.N+1]

        u_i_sub_1 = u_last[0:self.N-1]
        u_i = u_last[1:self.N]
        u_i_add_1 = u_last[2:self.N+1]
        # Calculate initial acceleration
        initial_a = np.zeros(self.N+1, np.float64)
        initial_a[1:self.N] = 1/self.dx**2 * (
            0.5*(c2_i_add_1 + c2_i)*(u_i_add_1-u_i)
            - 0.5*(c2_i+c2_i_sub_1)*(u_i-u_i_sub_1))
        # Calculate the waveform at t=1
        u_current[1:self.N] = u_last[1:self.N] + \
            initial_v[1:self.N] * self.dt +  \
            0.5 * initial_a[1:self.N] * self.dt**2
        # Apply boundary conditions
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
            # Calculate non-boundary points 1,...,N-1
            u_current_i_sub_1 = u_current[0:self.N-1]
            u_current_i = u_current[1:self.N]
            u_current_i_add_1 = u_current[2:self.N+1]
            u_last_i = u_last[1:self.N]

            u_next_i = 2*u_current_i - u_last_i + (self.dt/self.dx)**2*(
                0.5*(c2_i_add_1+c2_i)*(u_current_i_add_1-u_current_i)
                - 0.5*(c2_i + c2_i_sub_1)*(u_current_i - u_current_i_sub_1)
            )
            u_next[1:self.N] = u_next_i
            # Calculate boundary points
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
