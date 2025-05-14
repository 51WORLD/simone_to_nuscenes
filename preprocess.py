import utils
import shutil
import os
import numpy as np
import datetime
import json
import re

def extract_number(filename):
    match = re.search(r'(\d+).pcd', filename)
    if match:
        return int(match.group(1))
    return 0


def create_nuscenes_v1_0_structure(base_path):
    # Define the directory structure

    # Create the base directory
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Create the subdirectories
    for directory in utils.directories:
        dir_path = os.path.join(base_path, directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


def read_pcd_file(file_path):
    with open(file_path, 'rb') as f:
        lines = f.readlines()

        # 找到点数据的开始
        data_start_line = None
        for i, line in enumerate(lines):
            if line.startswith(b'DATA binary'):
                data_start_line = i + 1
                break
        else:
            raise ValueError('无效的PCD文件：未找到二进制点数据。')

        # 读取二进制数据
        data = b''.join(lines[data_start_line:])

        # 定义数据类型（包含所有字段）
        dtype = np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.uint32),
            ('intensity', np.uint8),
            ('segmentation', np.uint8),
            ('ring', np.uint8),
            ('angle', np.uint8)
        ])

        # 将二进制数据转换为结构化数组
        point_cloud = np.frombuffer(data, dtype=dtype)

        # 只保留 x, y, z, intensity
        point_cloud_filtered = np.zeros(point_cloud.shape[0], dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32), ('intensity', np.float32), ('ring', np.float32)])
        point_cloud_filtered['x'] = point_cloud['x']
        point_cloud_filtered['y'] = point_cloud['y']
        point_cloud_filtered['z'] = point_cloud['z']
        point_cloud_filtered['intensity'] = point_cloud['intensity']
        point_cloud_filtered['ring'] = point_cloud['ring']

        return point_cloud_filtered


if __name__ == "__main__":
    create_nuscenes_v1_0_structure(utils.base_path)
    case_root = 'simone_datasets'
    cases = os.listdir(case_root)
    # for case in cases:
    for index, case in enumerate(cases):
        print(index, case)
        timestart = utils.timestart + index * 5 * 60 * 1000000
        lidar_file_path = os.path.join(case_root, case, "lidar_top")
        print(lidar_file_path)
        dest_filepath_samples = 'nuscenes_v1.0/samples/LIDAR_TOP'
        dest_filepath_sweeps = 'nuscenes_v1.0/sweeps/LIDAR_TOP'

        if not os.path.exists(dest_filepath_samples):
            os.makedirs(dest_filepath_samples)

        if not os.path.exists(dest_filepath_sweeps):
            os.makedirs(dest_filepath_sweeps)

        formatted_date = utils.get_pcd_cretatime(lidar_file_path)
        dir_path = [fi for fi in os.listdir(lidar_file_path) if fi.endswith('.pcd')]
        dir_path = sorted(dir_path, key=extract_number)
        sample_counter = 0  # 用于跟踪何时将文件放入samples或sweeps

        for fi in dir_path:
            current_path = os.getcwd()
            lidar_path = os.path.join(current_path, lidar_file_path, fi)
            point_cloud = np.copy(read_pcd_file(lidar_path))

            # 保存为 .pcd.bin 文件
            save_lidar_path = fi.replace(".pcd", ".pcd.bin")
            filename = case + "-" + formatted_date + "-0800" + "__LIDAR_TOP__" + str(int(timestart + int(save_lidar_path.split('.')[0]) * 1 / 60 * 1000 * 1000)) + ".pcd.bin"

            # 根据sample_counter决定文件保存位置
            if sample_counter == 0:
                dest_filepath = dest_filepath_samples
                sample_counter = 9  # 下一个文件将进入sweeps，直到第10个
            else:
                dest_filepath = dest_filepath_sweeps
                sample_counter -= 1  # 每处理一个文件，计数器减1

            point_cloud.tofile(os.path.join(dest_filepath, filename))