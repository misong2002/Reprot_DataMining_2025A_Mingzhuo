#从hdf5中读出与particle physics有关的数据并储存
import numpy as np
import h5py
#命令行参数设置
import argparse
# -d --data_path: 数据文件路径
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', type=str, required=True, help='Path to the HDF5 data file')
#-o --output_path: 输出文件路径
parser.add_argument('-o', '--output_path', type=str, default='./data/physics_data.npz', help='Path to save the physics data HDF5 file')
args = parser.parse_args()

def generate_physics_data(data_path):
    #读取hdf5文件
    with h5py.File(data_path, 'r') as file:
        #读取events数据集
        events = file['events'][:]
    
    energy_prompt = events['energy_prompt_MeV']
    energy_delayed = events['energy_delayed_MeV']
    log_time_diff = np.log(events['delta_t_us'])
    #两个顶点的euclid距离
    vertex_prompt = np.vstack((events['vertex_prompt_x_mm'], events['vertex_prompt_y_mm'], events['vertex_prompt_z_mm'])).T
    vertex_delayed = np.vstack((events['vertex_delayed_x_mm'], events['vertex_delayed_y_mm'], events['vertex_delayed_z_mm'])).T
    log_vertex_distance =np.log(np.linalg.norm(vertex_prompt - vertex_delayed, axis=1)    ) 
    
    #定义一个新numpy array，包含上述四个物理量
    physics_data = np.zeros(events.shape, dtype=[('energy_prompt_MeV', 'f4'),
                                                ('energy_delayed_MeV', 'f4'),
                                                ('log_time_diff_us', 'f4'),
                                                ('log_vertex_distance_mm', 'f4')])
    physics_data['energy_prompt_MeV'] = energy_prompt
    physics_data['energy_delayed_MeV'] = energy_delayed
    physics_data['log_time_diff_us'] = log_time_diff
    physics_data['log_vertex_distance_mm'] = log_vertex_distance

    return physics_data

if __name__ == '__main__':
    physics_data = generate_physics_data(args.data_path)
    #保存为npz文件
    np.savez_compressed(args.output_path, physics_data=physics_data)
    print(f'Physics data saved to {args.output_path}')