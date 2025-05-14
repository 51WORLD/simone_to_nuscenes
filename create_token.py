import json
import os
from uuid import uuid4
import utils
import math
import re

# NUM_SCENE = utils.NUM_SCENE
# NUM_SENSOR = utils.NUM_SENSOR
# NUM_SAMPLE = utils.NUM_SAMPLE


# token_dict = {'scene': NUM_SCENE, 'sensor': NUM_SENSOR, 'sample': NUM_SAMPLE}

mapping_dict = {
    'front': 'CAM_FRONT',
    'front_left': 'CAM_FRONT_LEFT',
    'front_right': 'CAM_FRONT_RIGHT',
    'rear': 'CAM_BACK',
    'rear_left': 'CAM_BACK_LEFT',
    'rear_right': 'CAM_BACK_RIGHT',
    'lidar_top': 'LIDAR_TOP'
}



def extract_number(filename):
    match = re.search(r'(\d+).pcd', filename)
    if match:
        return int(match.group(1))
    return 0

# generate ego_pose
def generate_ego_pose_token():
    ego_pose_lst = []

    case_root = 'simone_datasets'
    cases = os.listdir(case_root)
    # for case in cases:
    for index, case in enumerate(cases):
        timestart = utils.timestart + index * 5 * 60 * 1000000
        ego_pose_pos_dict = {}
        ego_pose_rot_dict = {}
        for key, value in mapping_dict.items():
            # for camera. Actually timestamp of simone dataset including 6x camera and 1x lidar is same. Hence they share the same moment
            if key != 'lidar_top':
                key1 = os.path.join(case_root, case, key)
                files = [fi for fi in os.listdir(key1) if fi != 'DumpSettings.json']
                for idx, frame in enumerate(sorted(files, key=lambda x: int(x))):
                    with open(os.path.join(key1, frame, 'CameraInfo.json')) as f:
                        uuid = str(uuid4())
                        token = uuid.replace('-', '')
                        print(f)
                        data = json.load(f)
                        pos = data['ego_pos']
                        rot = data['ego_rot']
                        rot = utils.rot2fourvar(rot)
                        lidar_file_path = os.path.join(case_root, case, "lidar_top")
                        formatted_date = utils.get_pcd_cretatime(lidar_file_path)
                        filename = case + "-" + formatted_date + "-0800" + "__"+value+"__" + str(int(timestart + int(frame) * 1 / 60 * 1000 * 1000))

                        timestamp_lst = sorted([int(fi) for fi in os.listdir(key1) if fi != 'DumpSettings.json'],
                                               key=lambda x: int(x))
                        ego_pose_json = {
                            "filename": os.path.join(filename + '.jpg'),
                            "info": {
                                "token": token,
                                "timestamp": int(timestart + int(timestamp_lst[idx]) * 1 / 60 * 1000 * 1000),
                                "rotation": rot,
                                "translation": pos
                            }
                        }
                        ego_pose_lst.append(ego_pose_json)
                        # if frame not in ego_pose_rot_dict:
                        #     ego_pose_pos_dict[frame] = pos
                        #     ego_pose_rot_dict[frame] = rot



            else:
                key2 = os.path.join(case_root, case, key)
                files = [fi for fi in os.listdir(key2) if fi != 'DumpSettings.json' and fi.endswith('.pcd')]
                for idx, fi in enumerate(sorted(files, key=extract_number)):
                    match = re.search(r'(\d+)', fi)
                    if match:
                        number = int(match.group(1))
                    uuid = str(uuid4())
                    token = uuid.replace('-', '')
                    filename = case + "-" + formatted_date + "-0800" + "__" + value + "__" + str(int(timestart + number * 1 / 60 * 1000 * 1000))
                    number = str(number)

                    with open(os.path.join(key2, f'{number}.json')) as f:
                        print(f)
                        data = json.load(f)
                        pos = data['ego_pos']
                        rot = data['ego_rot']
                        rot = utils.rot2fourvar(rot)

                    ego_pose_json = {
                        "filename": os.path.join(filename + '.pcd'),
                        "info": {
                            "token": token,
                            "timestamp": int(timestart + int(number) * 1 / 60 * 1000 * 1000),
                            "rotation":rot,
                            "translation": pos
                        }
                    }
                    ego_pose_lst.append(ego_pose_json)

    json_file = 'data/ego_pose.json'
    with open(json_file, 'w') as f:
        json.dump(ego_pose_lst, f, indent=4)


# 生成UUID4并转换成nuScenes格式的token
def generate_token_list(num):
    lst = []
    for _ in range(num):
        uuid = str(uuid4())

        token = uuid.replace('-', '')
        lst.append(token)
    return lst


def generate_scene_json():
    sample_data_lst = []
    scene_json_lst = []
    case_root = 'simone_datasets'
    cases = os.listdir(case_root)
    for case in cases:
        test = list(mapping_dict.keys())
        front_cam = os.path.join(case_root, case, test[0])
        files = [f for f in os.listdir(front_cam) if f != 'DumpSettings.json']
        sample_data_lst.clear()
        counter = 0
        for idx, fi in enumerate(sorted(files, key=lambda x: int(x))):
            if counter % 6 == 0:
                counter += 1
                if fi != 'DumpSettings.json':
                    uuid = str(uuid4())
                    token = uuid.replace('-', '')
                    sample_data = {"samplename": fi, "token": token}
                    sample_data_lst.append(sample_data)
            else:
                counter += 1

        sample_obj = {"token": str(uuid4()).replace('-', ''), "samples": sample_data_lst.copy(),"scene-name": str(case)}
        scene_json_lst.append(sample_obj)
    return scene_json_lst


def generate_token_json():
    lst = []
    for key, value in mapping_dict.items():
        uuid = str(uuid4())
        token = uuid.replace('-', '')
        sensor_json = {"sensor_name": key, "token": token}
        lst.append(sensor_json)
    data = {
        "scene": generate_scene_json(),
        "sensor": lst
    }
    file_path = 'data/token.json'
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == '__main__':
    generate_ego_pose_token()
    generate_token_json()
