import json
import os
from uuid import uuid4
import utils

import math
import numpy as np

mapping_dict = {
    'front': 'CAM_FRONT',
    'front_left': 'CAM_FRONT_LEFT',
    'front_right': 'CAM_FRONT_RIGHT',
    'rear': 'CAM_BACK',
    'rear_left': 'CAM_BACK_LEFT',
    'rear_right': 'CAM_BACK_RIGHT',
    'lidar_top': 'LIDAR_TOP'
}

sensor_token = []
sensor_token_json_file = 'data/token.json'
with open(sensor_token_json_file, 'r') as f:
    sensor_data = json.load(f)["sensor"]
sensor_dict = {item["sensor_name"]: item["token"] for item in sensor_data}

def generate_calibrated_sensor_json():
    calibrated_sensor_lst=[]
    case_root = 'simone_datasets'
    cases = os.listdir(case_root)
    for case in cases:
        case_name_root = os.path.join(case_root, case)
        print(case_name_root)
        calibrated_sensor_per_scene_lst = []
        for key, value in mapping_dict.items():
            """
            for camera. Actually timestamp of simone dataset including 6x camera and 1x lidar is same. Hence they share the same moment
            From every sensor dataset folder,DumpSettings.json, camera_intrinsic including fx,fy,cx,cy ,and camera_extrinsic including pos and rpt
            
            camera intrinsic matrix
            K = [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
                ])
            """
            if key != 'lidar_top':
                sensor_root =os.path.join(case_name_root, key)
                with open(os.path.join(sensor_root, 'DumpSettings.json')) as f:
                    camera_data = json.load(f)["camera"]
                    pos = camera_data["pos"]
                    rot = camera_data["rot"]
                    fx = camera_data["fx"]
                    fy = camera_data["fy"]
                    cx = camera_data["cx"]
                    cy = camera_data["cy"]
                    # 构建相机内参矩阵 K
                    K = np.array([
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]
                    ])
                    # 生成唯一的 token
                    uuid = str(uuid4())
                    token = uuid.replace('-', '')

                    # 构建校准的传感器 JSON 数据
                    calibrated_sensor_json = {
                        "token": token,
                        "sensor_token": sensor_dict[key],  # 假设这是你的传感器 token 数据来源
                        "translation": pos,
                        "rotation": utils.transform_coordinates(rot,key).tolist(),  # 假设这是转换旋转表示的函数
                        "camera_intrinsic": K.tolist()  # 将 numpy 数组转换为 Python 原生列表
                    }
            else:
                sensor_root = os.path.join(case_name_root, key)
                with open(os.path.join(sensor_root, 'DumpSettings.json')) as f:
                    lidar_data = json.load(f)["lidar"]
                    pos = lidar_data["pos"]
                    rot = lidar_data["rot"]
                    uuid = str(uuid4())
                    token = uuid.replace('-', '')
                    calibrated_sensor_json = {
                        "token": token,
                        "sensor_token": sensor_dict[key],
                        "translation": list(pos),
                        "rotation": list(utils.rot2fourvar(rot)),
                        "camera_intrinsic": [],
                    }
            calibrated_sensor_lst.append(calibrated_sensor_json.copy())

    return calibrated_sensor_lst



def main():
    calibrated_sensor_lst = generate_calibrated_sensor_json()
    calibrated_sensor_json_file_path='data/calibrated_sensor.json'
    with open(calibrated_sensor_json_file_path,'w') as f:
        json.dump(calibrated_sensor_lst,f,indent=4)


if __name__ == '__main__':
    main()

