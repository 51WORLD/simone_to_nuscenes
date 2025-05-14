# 导入必要的库
import os
import numpy as np
import open3d as o3d
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

# 1. 加载 nuScenes 数据集
# 请将 'path_to_nuscenes_dataset' 替换为您实际的数据集路径
# nusc = NuScenes(version='v1.0-mini', dataroot='nuscenes-minilast', verbose=True)
nusc = NuScenes(version='v1.0-mini', dataroot='nuscenes_v1.0', verbose=True)
# 2. 定义加载点云数据的函数
def load_point_cloud(nusc, sample_data_token):
    # 获取点云文件的路径
    pcl_path = nusc.get_sample_data_path(sample_data_token)
    # 读取点云数据，包含 x, y, z, intensity
    point_cloud = np.fromfile(pcl_path, dtype=np.float32).reshape(-1, 5)[:, :4]
    return point_cloud


# 3. 定义加载 3D 激光雷达标注框的函数
# 3. 定义加载与激光雷达点云相关的 3D 标注框的函数
def load_lidar_boxes(nusc, sample_data_token):
    """
    只获取激光雷达点云的 3D 标注框
    :param nusc: NuScenes 对象
    :param sample_data_token: 激光雷达的 sample_data token
    :return: 返回所有 3D 标注框
    """
    # 获取点云对应的样本数据
    sample_data = nusc.get('sample_data', sample_data_token)
    sample = nusc.get('sample', sample_data['sample_token'])

    # 获取点云的传感器校准和自车位姿信息
    cs_record = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])

    # 获取与该帧激光雷达点云相关的所有标注框
    annotation_tokens = sample['anns']  # 只从样本中获取标注
    boxes = []

    # 遍历标注 token，并获取每个标注的 3D 边界框
    for ann_token in annotation_tokens:
        # 获取标注的 3D 边界框
        annotation = nusc.get('sample_annotation', ann_token)

        # 只获取与激光雷达点云相关的标注
        if annotation['num_lidar_pts'] > 0:
            box = Box(annotation['translation'],
                      annotation['size'],
                      Quaternion(annotation['rotation']),
                      name=annotation['category_name'])

            # 从全局坐标系转换到自车坐标系
            box.translate(-np.array(ego_pose['translation']))
            box.rotate(Quaternion(ego_pose['rotation']).inverse)

            # 从自车坐标系转换到传感器坐标系
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

            # 将转换后的标注框加入到列表中
            boxes.append(box)

    return boxes


# 3. 定义加载 3D 标注框的函数
def load_3d_boxes(nusc, sample_data_token):
    # 获取点云对应的样本数据
    sample_data = nusc.get('sample_data', sample_data_token)
    sample = nusc.get('sample', sample_data['sample_token'])

    # 获取点云的传感器校准和自车位姿信息
    cs_record = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])

    # 获取所有标注框
    boxes = nusc.get_boxes(sample_data_token)

    # 将标注框从全局坐标系转换到传感器坐标系
    for box in boxes:
        # 从全局坐标系转换到自车坐标系
        box.translate(-np.array(ego_pose['translation']))
        box.rotate(Quaternion(ego_pose['rotation']).inverse)

        # 从自车坐标系转换到传感器坐标系
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

    return boxes

# 4. 定义可视化函数，设置点的显示样式和大小
def visualize_point_cloud_with_boxes(point_cloud, boxes):
    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])

    # 获取强度值并进行自定义颜色映射
    intensity = point_cloud[:, 3]
    colors = []

    # 自定义颜色映射逻辑
    for i in intensity:
        if i <= 33:
            colors.append([0, int(7.727 * i), 255])
        elif i > 33 and i <= 66:
            colors.append([0, 255, int(255 - 7.727 * (i - 34))])
        elif i > 66 and i <= 100:
            colors.append([int(7.727 * (i - 67)), 255, 0])
        elif i > 100 and i <= 256:
            colors.append([255, int(255 - 7.727 * (i - 100) / 4.697), 0])

    # 将颜色列表转换为 numpy 数组并缩放到 [0, 1] 范围
    colors = np.array(colors).reshape(-1, 3) / 256.0

    # 设置点云的颜色
    pcd.colors = o3d.utility.Vector3dVector(colors)

    geometries = [pcd]

    # 为每个标注框创建 Open3D 几何体
    for box in boxes:
        corners = box.corners().T  # (8, 3)
        # 定义 12 条边连接点
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # 顶部边缘
            [4, 5], [5, 6], [6, 7], [7, 4],  # 底部边缘
            [0, 4], [1, 5], [2, 6], [3, 7]   # 竖直边缘
        ]
        # 设置边的颜色（红色）
        colors_box = [[1, 0, 0] for _ in lines]
        # 创建 LineSet 对象表示边框
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors_box)
        geometries.append(line_set)

    # 创建可视化窗口，并设置点的尺寸
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for geometry in geometries:
        vis.add_geometry(geometry)

    # 获取渲染选项并设置点的尺寸
    render_option = vis.get_render_option()
    render_option.point_size = 2.0  # 将点的尺寸设置为 1.0，数值越小，点越小
    render_option.background_color = np.array([0.0, 0.0, 0.0])

    # 启动可视化
    vis.run()
    # vis.destroy_window()

# 5. 选择一个样本进行可视化
# 这里选择第一个场景的第一个样本，您可以根据需要修改
scene = nusc.scene[1]
first_sample_token = scene['first_sample_token']
while first_sample_token:
    sample = nusc.get('sample', first_sample_token)
    lidar_token = sample['data']['LIDAR_TOP']

    # 加载点云数据
    point_cloud = load_point_cloud(nusc, lidar_token)

    # 加载 3D 标注框
    boxes = load_lidar_boxes(nusc, lidar_token)
    # 可视化
    visualize_point_cloud_with_boxes(point_cloud, boxes)
    first_sample_token = sample['next']
# 获取激光雷达数据的 sample_data_token

