# 从 hdf5 中读出与 particle physics 有关的数据并储存为 npz
import argparse
import numpy as np
import h5py


# ----------------------------
# Argument parsing
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data_path',
        type=str,
        required=True,
        help='Path to the HDF5 data file'
    )
    parser.add_argument(
        '-o', '--output_path',
        type=str,
        default='./physics_data.npz',
        help='Path to save the physics data npz file'
    )
    return parser.parse_args()


# ----------------------------
# Core data processing
# ----------------------------
def generate_physics_data(data_path):
    # 直接用只读方式打开，避免不必要的 copy
    with h5py.File(data_path, 'r') as f:
        events = f['events'][:]

    # 选择 n_det ≈ 8 的事件
    mask = np.abs(events['n_det'] - 8) < 0.1
    events = events[mask]

    # -------- 能量 --------
    energy_prompt = events['energy_prompt_MeV'].astype(np.float32)
    energy_delayed = events['energy_delayed_MeV'].astype(np.float32)

    # -------- 时间差（log）--------
    log_time_diff_us = np.log(events['delta_t_us']).astype(np.float32)

    # -------- 顶点距离（log）--------
    vertex_prompt = np.stack(
        (
            events['vertex_prompt_x_mm'],
            events['vertex_prompt_y_mm'],
            events['vertex_prompt_z_mm'],
        ),
        axis=1
    )

    vertex_delayed = np.stack(
        (
            events['vertex_delayed_x_mm'],
            events['vertex_delayed_y_mm'],
            events['vertex_delayed_z_mm'],
        ),
        axis=1
    )

    log_vertex_distance_mm = np.log(
        np.linalg.norm(vertex_prompt - vertex_delayed, axis=1)
    ).astype(np.float32)

    return {
        'energy_prompt_MeV': energy_prompt,
        'energy_delayed_MeV': energy_delayed,
        'log_time_diff_us': log_time_diff_us,
        'log_vertex_distance_mm': log_vertex_distance_mm,
    }


# ----------------------------
# Main
# ----------------------------
if __name__ == '__main__':
    args = parse_args()

    physics_data = generate_physics_data(args.data_path)

    # 直接把字段摊平存到 npz（一级目录）
    np.savez_compressed(args.output_path, **physics_data)

    print(f'Physics data saved to {args.output_path}')
