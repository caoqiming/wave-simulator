import numpy as np
import abc
from typing import Callable, Any
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class BoundaryCondition(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def apply(self, u_0: np.float64, u_1: np.float64, **kwargs: Any) -> np.float64:
        """
        以左边界为例，输入上一时刻的 u_0, u_1，输出下一时刻的 u_0
        """
        raise RuntimeError("not implemented")

    @abc.abstractmethod
    def apply2D(self, u_0_j: np.typing.NDArray, u_1_j: np.typing.NDArray, **kwargs: Any) -> np.typing.NDArray:
        """
        以左边界为例，输入上一时刻的 u_0_j, u_1_j，输出下一时刻的 u_0_j
        """
        raise RuntimeError("not implemented")


class FixedBoundary(BoundaryCondition):
    """
    固定边界
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
    边界点的位移梯度（斜率）为零（例如，一根绳子的末端系在一个可以在杆上自由滑动的环上）。
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
        # 先处理中间的点
        u_0_j_next[1:N] = 2*u_0_j[1:N] - u_0_j_last[1:N] + C2[1:N] * \
            (u_1_j[1:N]*2 + u_0_ja1 + u_0_js1 - 4*u_0_j[1:N])
        # 角落的点不知道怎么处理，角落的点属于两个边，脑瓜疼
        # 就先假定它属于的另一条边也是同样的 NeumannBoundary 吧，具体实现的时后面的优先级更高。
        u_0_j_next[0] = 2*u_0_j[0] - u_0_j_last[0] + C2[0] * \
            (u_1_j[0]*2 + 2*u_0_j[1] - 4*u_0_j[0])
        u_0_j_next[N] = 2*u_0_j[N] - u_0_j_last[N] + C2[N] * \
            (u_1_j[N]*2 + 2*u_0_j[N-1] - 4*u_0_j[N])
        return u_0_j_next


class UnlimitedBoundary(BoundaryCondition):
    """
    无限制边界
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


class TwoDimensionSimulator:
    def __init__(self):
        # 空间
        self.L_x = 5  # 仿真的距离范围，从 0 到 L_x
        self.dx = 0.05  # 仿真的最小距离间隔
        self.N_x = int(self.L_x/self.dx)  # 仿真的线段的个数
        self.X = np.linspace(0, self.L_x, self.N_x+1,
                             dtype=np.float64)  # 仿真的空间范围
        #
        self.L_y = 5
        self.dy = self.dx  # 强制dy=dx
        self.N_y = int(self.L_y/self.dy)
        self.Y = np.linspace(0, self.L_y, self.N_y+1,
                             dtype=np.float64)

        # 时间
        self.L_t = 8  # Duration of simulation [s]
        self.dt = 0.005  # 时间间隔
        self.N_t = int(self.L_t/self.dt)  # 时间段的个数
        self.T = np.linspace(0, self.L_t, self.N_t+1)  # 时间范围

        # 边界条件
        self.left_boundary = FixedBoundary()
        self.right_boundary = FixedBoundary()
        self.up_boundary = FixedBoundary()
        self.down_boundary = FixedBoundary()

        # 初始波形，默认全为 0
        self.initial_wave = lambda x, y: 0.0
        # 每个质点的初始速度，默认为 0
        self.initial_point_speed = lambda x, y: 0.0
        # 介质波速，默认为 1
        self.wave_speed = lambda x, y: 1.0

    def set_space_range(self, L_x, dx):
        self.L_x = L_x  # 仿真的距离范围，从 0 到 L_x
        self.dx = dx  # 仿真的最小距离间隔
        self.N_x = int(self.L_x/self.dx)  # 仿真的线段的个数
        self.X = np.linspace(0, self.L_x, self.N_x+1,
                             dtype=np.float64)  # 仿真的空间范围

    def set_time_range(self, L_t, dt):
        # 时间
        self.L_t = L_t  # Duration of simulation [s]
        self.dt = dt  # 时间间隔
        self.N_t = int(self.L_t/self.dt)  # 时间段的个数
        self.T = np.linspace(0, self.L_t, self.N_t+1)  # 时间范围

    def set_initial_wave(
        self,
        initial_wave: Callable[[np.float64, np.float64], np.float64],
    ):
        """
        设置初始波形。用一个函数表示，输入空间的横坐标x，返回对应位置的波函数值
        """
        self.initial_wave = initial_wave

    def set_initial_point_speed(
        self,
        initial_point_speed: Callable[[np.float64, np.float64], np.float64],
    ):
        """
        设置每个质点的初始速度。用一个函数表示，输入空间的横坐标 x 和纵坐标 y，返回对应位置质点的初始速度
        """
        self.initial_point_speed = initial_point_speed

    def set_wave_speed(
        self,
        wave_speed: Callable[[np.float64, np.float64], np.float64],
    ):
        """
        设置介质中的波速。用一个函数表示，输入空间的横坐标 x 和纵坐标 y，返回对应位置介质中的波速。
        """
        self.wave_speed = wave_speed

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
        开始仿真
        """
        # 用于储存结果
        self.result = np.zeros(
            (self.N_x+1, self.N_y+1, self.N_t+1), np.float64)

        # 用于储存当前 i-1,i,i+1 时刻的波形
        u_last = np.zeros((self.N_x+1, self.N_y+1), np.float64)
        u_current = np.zeros((self.N_x+1, self.N_y+1), np.float64)
        u_next = np.zeros((self.N_x+1, self.N_y+1), np.float64)

        # 初始化
        c = np.zeros((self.N_x+1, self.N_y+1), np.float64)
        initial_v = np.zeros((self.N_x+1, self.N_y+1), np.float64)
        for i in range(0, self.N_x+1):
            for j in range(0, self.N_y+1):
                # 介质中的波速（注意 Y 的索引应为 j）
                c[i, j] = self.wave_speed(self.X[i], self.Y[j])
                # 初始波形
                u_last[i, j] = self.initial_wave(self.X[i], self.Y[j])
                # 初始质点速度
                initial_v[i, j] = self.initial_point_speed(
                    self.X[i], self.Y[j])

        # 定义一些常数避免重复计算
        c2 = c**2
        C = c*self.dt/self.dx
        C2 = C**2

        initial_a = np.zeros((self.N_x+1, self.N_y+1), np.float64)

        # 对于二维的仿真，我们用 i 表示 空间x 的索引，j 表示 空间y 的索引，k 表示时间的索引
        # 那么 i,j 位置为 [1:self.N_x,1:self.N_y]
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

        # 计算初始加速度
        initial_a[1:self.N_x, 1:self.N_y] = c2_i_j/self.dx**2 * (
            u_ip1_j + u_is1_j + u_i_ja1 + u_i_js1 - 4*u_i_j
        )
        # 计算 t=1 时刻的波形
        u_current[1:self.N_x, 1:self.N_y] = u_i_j + \
            initial_v[1:self.N_x, 1:self.N_y] * self.dt +\
            0.5*initial_a[1:self.N_x, 1:self.N_y]*self.dt**2
        # 应用边界条件
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
            # 计算非边界的点 1,...,N-1
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
            # 计算边界的点
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


def animate_result_3d(result_matrix, X, Y,  cmap='viridis',
                      z_label="U", title_prefix="3D wave", fps=60, interval=0.05,
                      save_path=None):
    # 1. 从一维 X 和 Y 数组获取尺寸
    Nx = len(X)
    Ny = len(Y)

    # 2. 检查 result_matrix 的形状
    if result_matrix.ndim != 3:
        raise ValueError(f"result_matrix 必须是三维的，但其维度为 {result_matrix.ndim}。")

    Ny_data, Nx_data, Nt = result_matrix.shape

    # 3. 检查 result_matrix 的空间维度是否与 X 和 Y 的长度匹配
    if Ny_data != Ny or Nx_data != Nx:
        raise ValueError(
            f"result_matrix 的形状 ({Ny_data}, {Nx_data}, {Nt}) 与 X, Y 数组的长度不匹配。\n"
            f"期望的空间维度应为 ({Ny}, {Nx})"
        )

    # 4. 生成用于绘图的二维网格
    # 注意：meshgrid 默认生成 (Ny, Nx) 形状的网格，这与 result_matrix 的前两维顺序 (y, x) 匹配
    X_grid, Y_grid = np.meshgrid(X, Y)

    # --- 绘图初始化 ---

    # 创建一个图形和三维子图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 初始绘图（第一帧）
    # 绘制一个曲面图
    surf = ax.plot_surface(
        X_grid, Y_grid, result_matrix[:, :, 0], cmap=cmap, edgecolor='none')

    # 设置轴标签和标题
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel(z_label)

    # 根据数据范围设置Z轴的固定限制
    z_min = np.min(result_matrix)
    z_max = np.max(result_matrix)
    ax.set_zlim(z_min, z_max)

    # 添加颜色条
    fig.colorbar(surf, shrink=0.5, aspect=5, label=z_label)

    # --- 动画更新函数 ---

    def update(frame):
        ax.cla()  # 清除当前轴内容

        # 重新绘制曲面，使用当前时间步的数据
        surf = ax.plot_surface(
            X_grid, Y_grid, result_matrix[:, :, frame], cmap=cmap, edgecolor='none')

        # 重新设置轴标签和 Z 轴限制
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel(z_label)
        ax.set_zlim(z_min, z_max)

        ax.set_title(f"{title_prefix}: {frame+1}/{Nt}")

        return surf,

    # --- 创建动画 ---

    ani = FuncAnimation(fig, update, frames=Nt, interval=interval, blit=False)
    if save_path is not None:
        try:
            print("Saving, this may take a while...")
            ani.save(
                filename=save_path,  # 文件名
                writer='ffmpeg',                    # 明确指定使用 ffmpeg 写入器
                dpi=150,
                fps=fps,
            )
            print(f"saved to {save_path}")

        except ValueError as e:
            # 捕获可能由 ffmpeg 找不到引起的错误
            if "Requested MovieWriter (ffmpeg) not available" in str(e):
                print("\n=== 错误提示 ===")
                print("无法找到 FFmpeg 库！要保存为 .mp4 格式，您需要先安装 FFmpeg。")
    plt.tight_layout()
    plt.show()
    return ani


def animate_result_flat(result_matrix, X, Y,
                        interval=1,
                        cmap='viridis',
                        vmin=None, vmax=None,
                        downsample_temporal: int = 1,
                        show: bool = True,
                        save_path: str = None,
                        fps=60,
                        ):
    if result_matrix.ndim != 3:
        raise ValueError("result_matrix 必须为三维数组 (Nx, Ny, Nt) 或 (Ny, Nx, Nt)")

    Nx = len(X)
    Ny = len(Y)
    a, b, Nt = result_matrix.shape

    # 兼容两种轴顺序：如果 (Ny, Nx, Nt) 则直接使用；如果 (Nx, Ny, Nt) 则转置为 (Ny, Nx, Nt)
    if (a == Ny and b == Nx):
        data = result_matrix
    elif (a == Nx and b == Ny):
        data = np.transpose(result_matrix, (1, 0, 2))
    else:
        raise ValueError(
            f"result_matrix 的空间维度与 X,Y 长度不匹配：got {a,b}, expected ({Ny},{Nx}) or ({Nx},{Ny})")

    # 根据 downsample_temporal 选择帧索引
    if downsample_temporal > 1:
        frame_indices = list(range(0, data.shape[2], downsample_temporal))
    else:
        frame_indices = list(range(data.shape[2]))

    extent = [float(X.min()), float(X.max()), float(Y.min()), float(Y.max())]

    fig, ax = plt.subplots()
    im = ax.imshow(data[:, :, frame_indices[0]], origin='lower', extent=extent,
                   cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    fig.colorbar(im, ax=ax)
    title = ax.set_title(f"t=0 (frame 0/{len(frame_indices)})")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    def _update_frame(idx):
        k = frame_indices[idx]
        im.set_data(data[:, :, k])
        title.set_text(f"frame {idx+1}/{len(frame_indices)} (t={k})")
        return [im]

    ani = FuncAnimation(fig, _update_frame, frames=len(
        frame_indices), interval=interval, blit=True)

    if save_path is not None:
        try:
            print("Saving, this may take a while...")
            ani.save(
                filename=save_path,  # 文件名
                writer='ffmpeg',                    # 明确指定使用 ffmpeg 写入器
                dpi=150,
                fps=fps,
            )
            print(f"saved to {save_path}")

        except ValueError as e:
            # 捕获可能由 ffmpeg 找不到引起的错误
            if "Requested MovieWriter (ffmpeg) not available" in str(e):
                print("\n=== 错误提示 ===")
                print("无法找到 FFmpeg 库！要保存为 .mp4 格式，您需要先安装 FFmpeg。")

    if show:
        plt.tight_layout()
        plt.show()

    return ani


if __name__ == "__main__":
    s = TwoDimensionSimulator()

    def I(x, y):
        """
        two space variables depending function 
        that represent the wave form at t = 0
        """
        return 0.2*np.exp(-((x-1)**2/0.1 + (y-1)**2/0.1))
    s.set_initial_wave(I)

    s.set_right_boundary(NeumannBoundary())
    s.set_left_boundary(NeumannBoundary())
    s.set_up_boundary(NeumannBoundary())
    s.set_down_boundary(NeumannBoundary())

    s.simulate()
    # s.result is (Nx+1, Ny+1, Nt+1)
    # 在 notebook 中显示动画（也可以把返回值保存为 anim）
    plt.rcParams['animation.embed_limit'] = 1000.0
    # animate_result_3d(s.result, X=s.X, Y=s.Y, interval=0.01)
    # 也可以选择平面动画
    animate_result_flat(s.result, X=s.X, Y=s.Y,
                        interval=0.01, fps=60, save_path="flat.mp4")
