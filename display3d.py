import cv2
import math
import os
import numpy as np


SegmentationColorTable = np.array([
    [0, 0, 0], [107, 142, 35], [70, 70, 70], [128, 64, 128], [220, 20, 60],
    [153, 153, 153], [0, 0, 142], [0, 0, 0], [119, 11, 32], [190, 153, 153],
    [70, 130, 180], [244, 35, 232], [240, 240, 240], [220, 220, 0], [102, 102, 156],
    [250, 170, 30], [152, 251, 152], [255, 0, 0], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [111, 74, 0], [180, 165, 180], [81, 0, 81],
    [150, 100, 100], [220, 220, 0], [169, 11, 32], [250, 170, 160], [230, 150, 140],
    [150, 120, 90], [151, 124, 0], [70, 120, 120], [70, 12, 120], [70, 120, 12],
    [0, 120, 120], [200, 120, 120], [70, 200, 120], [70, 120, 200], [100, 0, 0],
    [250, 120, 120], [70, 0, 250], [140, 100, 100], [160, 160, 160], [170, 10, 10],
    [130, 100, 10], [170, 100, 10], [170, 10, 100], [170, 170, 170], [100, 20, 10]
])

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

        # 定义数据类型
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

        return point_cloud

def intensity_to_rgb(intensity):
    if intensity <= 33:
        r = 0
        g = int(7.727 * intensity)
        b = 255
    elif 33 < intensity <= 66:
        r = 0
        g = 255
        b = int(255 - 7.727 * (intensity - 34))
    elif 66 < intensity <= 100:
        r = int(7.727 * (intensity - 67))
        g = 255
        b = 0
    elif 100 < intensity <= 255:
        r = 255
        g = int(255 - 7.727 * (intensity - 100) / 4.697)
        b = 0
    else:
        r, g, b = 255, 255, 255  # 默认值

    return r,g,b,255  # 返回 RGB 颜色以及 alpha 通道的值

def drawPcdtoImage(img, points, intensitys, segmentations, type=1):
    points = np.array(points, dtype=int)
    intensitys = np.array(intensitys, dtype=np.uint32)
    segmentations = np.array(segmentations, dtype=np.uint32)

    valid_mask = (points[:, 0] >= 0) & (points[:, 0] < img.shape[1]) & \
                 (points[:, 1] >= 0) & (points[:, 1] < img.shape[0])
    points = points[valid_mask]
    intensitys = intensitys[valid_mask]
    segmentations = segmentations[valid_mask]

    if type == 0:
        colors = np.array([intensity_to_rgb(intensity) for intensity in intensitys])
    elif type == 1:
        colors = SegmentationColorTable[segmentations]

    for (x, y), (red,green,blue, *_) in zip(points, colors):
        cv2.circle(img, (x, y), 2, (int(blue), int(green), int(red)), -1)

def is_in_fov(points, camera_matrix):
    fx = camera_matrix[0, 0]
    cx = camera_matrix[0, 2]
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    condition1 = z > 0
    test = np.arctan(cx / fx)
    condition2 = np.arctan(-abs(x) / z) < np.arctan(cx / fx)
    return condition1 & condition2

if __name__ == "__main__":
    lidar_file_path='lidar_top'
    dest_filepath='nuscenes_v1.0/samples/LIDAR_TOP'
    if not os.path.exists(dest_filepath):
        os.makedirs(dest_filepath)
    dir_path=[fi for fi in os.listdir(lidar_file_path) if fi.endswith('pcd')]
    for fi in dir_path:
        current_path = os.getcwd()
        lidar_path = os.path.join(current_path, "lidar_top",fi)
        point_cloud = np.copy(read_pcd_file(lidar_path))
        save_lidar_path = fi.replace(".pcd", ".pcd.bin")
        point_cloud.tofile(os.path.join(dest_filepath,save_lidar_path))

