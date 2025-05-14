import os
import json
import yaml
import math
import numpy as np
import datetime

import transforms3d.euler as euler

yaml_path = "simone.yaml"
datasets_path = 'datasets'
# Load data from token.yaml
with open(yaml_path, 'r') as f:
    yaml_data = yaml.safe_load(f)

simone2nuscenes_lst = [4, 6, 8, 17, 18, 19, 20, 21, 27]
NUM_SCENE = yaml_data["INFO"]["NUM_SCENE"]
NUM_SENSOR = yaml_data["INFO"]["NUM_SENSOR"]
NUM_SAMPLE = yaml_data["INFO"]["NUM_SAMPLE"]
FORMAT = yaml_data["INFO"]["FORMAT"]
WIDTH = yaml_data["INFO"]["WIDTH"]
HEIGHT = yaml_data["INFO"]["HEIGHT"]
NUM_CAM = yaml_data["INFO"]["NUM_CAM"]
NUM_LIDAR = yaml_data["INFO"]["NUM_LIDAR"]
NUM_RADAR = yaml_data["INFO"]["NUM_RADAR"]


point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]


timestart = 1533151603512000

assert NUM_SENSOR == NUM_CAM + NUM_LIDAR + NUM_RADAR, "NUM_SENSOR must equal with NUM_CAM+NUM_LIDAR+NUM_RADAR in simone.yaml"

base_path = 'nuscenes_v1.0'
directories = [
    'samples',
    'sweeps',
    'maps',
    'v1.0-mini'
]

from scipy.spatial.transform import Rotation as R


def rot2fourvar(matrix):
    # r = R.from_euler('xyz', euler, degrees=False)
    # quaternion = r.as_quat()
    # return quaternion.tolist()
    x_new = matrix[0]
    y_new = matrix[1]
    z_new = matrix[2]
    return euler.euler2quat(*list([x_new, y_new, z_new])).tolist()


def convert_pos_from_simone2nuscnes(p_original):
    p_target = [p_original[0], p_original[1], p_original[2]]
    return p_target

def get_pcd_cretatime(lidar_info_path):
    lidar_info_path = os.path.join(lidar_info_path, 'DumpSettings.json')
    setting_file = open(lidar_info_path, "rb")
    settingFileJson = json.load(setting_file)
    create_time = settingFileJson["createTime"]
    dt_object = datetime.datetime.fromtimestamp(create_time)
    formatted_date = dt_object.strftime('%Y-%m-%d-%H-%M-%S')
    return formatted_date

# bias_calibrated_sensor_dict = {
#     'front': -1.5522055408530322,
#     'front_left': -1.5441167875724489,
#     'front_right': -1.5602130619077934,
#     'rear': -1.54,
#     'rear_left': -1.5625751836932054,
#     'rear_right': -1.5628318096196687
#
# }


def transform_coordinates(matrix, key):
    x_new = -1.57
    y_new = matrix[1]
    z_new = matrix[2] - 1.57
    return euler.euler2quat(*list([x_new, y_new, z_new]))


if __name__ == '__main__':
    a = [0, 0, 0]
    print(rot2fourvar(a))
    print(np.array(euler.quat2euler(b)) * (180 / 3.14))
