import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

plt.rcParams['animation.embed_limit'] = 1000.0


def animate_result_1D(result, x, dt, down_sampling_rate=10, fps=60,
                      xlim=(0, 4), ylim=(-4, 4), save_path=None):

    fig = plt.figure()
    ax = plt.axes(xlim=xlim, ylim=ylim)
    line, = ax.plot([], [])
    ax.set_xlabel("x [m]", fontname="serif", fontsize=14)
    ax.set_ylabel("$u$ [m]", fontname="serif", fontsize=14)

    def init():
        line.set_data([], [])
        return line,

    # animation function.  This is called sequentially
    def animate(i):
        line.set_data(x, result[:, down_sampling_rate*i])
        return line,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=result.shape[1]//down_sampling_rate, interval=10, blit=True)

    if save_path is not None:
        try:
            print("Saving, this may take a while...")
            anim.save(
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
    return anim


def animate_result_3D(result_matrix, X, Y,  cmap='viridis',
                      z_label="U", title_prefix="3D wave", fps=60, interval=1,
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
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
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
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
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

    data = np.transpose(result_matrix, (1, 0, 2))

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
