import json
import os
from uuid import uuid4
import utils
import numpy as np
from uuid import uuid4
import copy
import re

def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

mapping_dict = {
    'front': 'CAM_FRONT',
    'front_left': 'CAM_FRONT_LEFT',
    'front_right': 'CAM_FRONT_RIGHT',
    'rear': 'CAM_BACK',
    'rear_left': 'CAM_BACK_LEFT',
    'rear_right': 'CAM_BACK_RIGHT',
    'lidar_top': 'LIDAR_TOP'
}

attributed_json_filepath = 'data/attribute.json'
with open(attributed_json_filepath, "r") as f:
    attributed_data = json.load(f)
attributed_name_with_token = {item["name"]: item["token"] for item in attributed_data}

simone_attribute_dict = {

    '6': ['vehicle.moving', 'vehicle.stopped'],  # Car
    '18': ['vehicle.moving', 'vehicle.stopped'],  # Truck
    '19': ['vehicle.moving', 'vehicle.stopped'],  # BUs
    '20': ['vehicle.moving', 'vehicle.stopped'],  # SpecialVehicle

    '8': ['vehicle.moving', 'vehicle.stopped'],  # Bicycle
    '21': ['vehicle.moving', 'vehicle.stopped'],  # Motorcycle
    '17': ['vehicle.moving', 'vehicle.stopped'],  # Rider
    '27': ['vehicle.moving', 'vehicle.stopped'],  # StaticBicycle

    '4': ['pedestrian.moving', 'pedestrian.standing']  # Pedestrian
}


def determine_attribute_status(type, vel):
    attributed_token = []
    current_attribute = simone_attribute_dict[str(type)][0] if any(vel) else simone_attribute_dict[str(type)][1]
    if type in [6, 18, 19, 20]:
        attributed_token.append(attributed_name_with_token[current_attribute])
    elif type in [8, 21, 27]:
        attributed_token.append(attributed_name_with_token[current_attribute])
    elif type in [17]:
        attributed_token.append(attributed_name_with_token["cycle.with_rider"])
    elif type in [4]:
        attributed_token.append(attributed_name_with_token[current_attribute])
    return attributed_token


def determine_grade(score):
    bins = np.array([0, 40, 60, 80, 100]) / 100
    labels = [
        1, 2, 3, 4
    ]

    grade = np.digitize(score, bins, right=True)
    return labels[grade - 1]


def get_sample_annotation(token, sample_token, instance_token, visibility_token, attribute_tokens, translation, size,
                          rotation, prev, Next, num_lidar_pts, num_radar_pts):
    sample_annotation = {
        "token": token,
        "sample_token": sample_token,
        "instance_token": instance_token,
        "visibility_token": visibility_token,
        "attribute_tokens":
            attribute_tokens,
        "translation": translation,
        "size": size,
        "rotation": rotation,
        "prev": prev,
        "next": Next,
        "num_lidar_pts": num_lidar_pts,
        "num_radar_pts": num_radar_pts
    }
    return sample_annotation


def calculate_annotation():

    case_root = 'simone_datasets'
    cases = os.listdir(case_root)
    num_annotation, dict_by_sample = 0, {}
    dict_by_id = {}
    dict_id={}
    for case in cases:
        for key, value in mapping_dict.items():
            sensor_path = os.path.join(case_root, case, key)
            if key != 'lidar_top':
                files = [fi for fi in os.listdir(sensor_path) if fi != 'DumpSettings.json']
                for idx, fi in enumerate(sorted(files, key=lambda x: int(x))):
                    with open(os.path.join(sensor_path, fi, 'CameraInfo.json'), "r") as f:
                        data = json.load(f)["bboxes3D"]
                    for item in data:
                        type = item["type"]
                        pos_ego_car = item["Pos_EgoCar"]
                        if not is_within_range(pos_ego_car, utils.point_cloud_range):
                            # print("++++++++++++++++++++++++++++++++++++++++++++++")
                            continue
                        if type in utils.simone2nuscenes_lst:
                            id = item["id"]
                            id_frame_case = case+"_"+fi+"_"+str(id)
                            id_case = case+"_"+str(id)
                            if id_case not in dict_id:
                                uuid = str(uuid4())
                                token = uuid.replace('-', '')
                                dict_id[id_case] = token
                            if id_frame_case not in dict_by_id:
                                dict_by_id[id_frame_case] = {}
                                dict_by_id[id_frame_case]["nbr_annotations"] = 0
                                dict_by_id[id_frame_case]["token"] = dict_id[id_case]
                                dict_by_id[id_frame_case]["annotation"] = []
                                dict_by_id[id_frame_case]["type"] = type
                            if dict_by_id[id_frame_case]["nbr_annotations"]==0:
                                dict_by_id[id_frame_case]["nbr_annotations"] += 1
            else:
                sensor_path = os.path.join(case_root, case, key)
                files = [fi.split('.')[0] for fi in os.listdir(sensor_path) if not fi.endswith('json')]
                for idx, fi in enumerate(sorted(files, key=lambda x: int(x))):
                    with open(os.path.join(sensor_path, f"{fi}.json"), "r") as f:
                        ann = json.load(f)
                        data = ann["bboxes3D"]
                        pos_ego = ann["pos"]
                        for item in data:
                            type = item["type"]
                            if type in utils.simone2nuscenes_lst:
                                id = item["id"]
                                id_frame_case = case+"_"+fi+"_"+str(id)
                                pos = [item["pos"][0], item["pos"][1], item["pos"][2]]
                                if id_case not in dict_id:
                                    uuid = str(uuid4())
                                    token = uuid.replace('-', '')
                                    dict_id[id_case] = token
                                pos_n = np.array(pos)
                                pos_ego_n = np.array(pos_ego)
                                relpos = pos_n - pos_ego_n
                                if not is_within_range(relpos, utils.point_cloud_range):
                                    continue
                                if id_frame_case not in dict_by_id:
                                    uuid = str(uuid4())
                                    token = uuid.replace('-', '')
                                    dict_by_id[id_frame_case] = {}
                                    dict_by_id[id_frame_case]["nbr_annotations"] = 0
                                    dict_by_id[id_frame_case]["token"] = dict_id[id_case]
                                    dict_by_id[id_frame_case]["annotation"] = []
                                    dict_by_id[id_frame_case]["type"] = type
                                if dict_by_id[id_frame_case]["nbr_annotations"] == 0:
                                    dict_by_id[id_frame_case]["nbr_annotations"] += 1

    for key, value in dict_by_id.items():

        nbr = value['nbr_annotations']
        print("--------------",key,nbr)
        for _ in range(nbr):
            uuid = str(uuid4())
            token = uuid.replace('-', '')
            dict_by_id[key]["annotation"].append(token)
    # 保存instances 到 instances.json
    instances_token_filepath = "data/instances.json"
    with open(instances_token_filepath, "w") as f:
        json.dump(dict_by_id, f, indent=4)
    return dict_by_id


def is_within_range(relpos, point_cloud_range):
    """
    判断 relpos 是否在给定的点云范围内
    :param relpos: NumPy 数组，表示相对位置 [x, y, z]
    :param point_cloud_range: 点云范围 [x_min, y_min, z_min, x_max, y_max, z_max]
    :return: 布尔值，True 表示在范围内，False 表示不在范围内
    """
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range

    # 检查 x, y, z 是否在范围内
    in_x_range = x_min <= relpos[0] <= x_max
    in_y_range = y_min <= relpos[1] <= y_max
    in_z_range = True

    # 返回所有条件的逻辑与
    return in_x_range and in_y_range and in_z_range

def main():
    dict_by_id = calculate_annotation()
    sample_annotation_json_lst = []
    idx_per_id = {k: 0 for k, v in dict_by_id.items()}

    case_root = 'simone_datasets'
    cases = os.listdir(case_root)
    num_annotation, dict_by_sample = 0, {}

    with open("data/token.json", "r") as f:
        sample_token_lst = json.load(f)["scene"]
    for sample_token, case in zip(sample_token_lst, cases):

        sample_dict = {si["samplename"]: si["token"] for si in sample_token["samples"]}
        for key, value in mapping_dict.items():
            sensor_path = os.path.join(case_root,case,key)
            if key != 'lidar_top':
                files = [fi for fi in os.listdir(sensor_path) if fi != 'DumpSettings.json']
                for idx, fi in enumerate(sorted(files, key=lambda x: int(x))):
                    # change area
                    if fi not in sample_dict:
                        continue
                    with open(os.path.join(sensor_path, fi, 'CameraInfo.json'), "r") as f:
                        data = json.load(f)
                        TwoDBBOXES = data["bboxes"]
                        TwoDBBOXESCULLED = data["bboxesCulled"]
                        ThreeDBBOXES = data["bboxes3D"]
                        TwoDBBOXES_SIZE = len(TwoDBBOXES)
                    for idx, item in enumerate(ThreeDBBOXES):
                        type = item["type"]
                        pos_ego_car = item["Pos_EgoCar"]
                        if not is_within_range(pos_ego_car, utils.point_cloud_range):
                            # print("++++++++++++++++++++++++++++++++++++++++++++++")
                            continue
                        if type in utils.simone2nuscenes_lst:
                            pos = [item["pos"][0], item["pos"][1], item["pos"][2]]
                            rot = item["rot"]
                            size = [item["size"][1], item["size"][0], item["size"][2]]
                            id = item["id"]
                            id_frame_case = case+"_"+fi+"_"+str(id)
                            # pre_id_frame_case = [str(case + "_" + str(int(fi_tmp)) + "_" + str(id)) in dict_by_id  for fi_tmp in range(int(fi) - 3, -1, -3)]

                            pre_id_frame_case = ""
                            for fi_tmp in range(int(fi) - 30, -1, -30):
                                candidate = f"{case}_{fi_tmp}_{id}"
                                if candidate in dict_by_id:
                                    pre_id_frame_case = candidate
                                    break

                            next_id_frame_case = ""
                            for fi_tmp in range(int(fi) + 30, 2000, 30):
                                candidate = f"{case}_{fi_tmp}_{id}"
                                if candidate in dict_by_id:
                                    next_id_frame_case = candidate
                                    break
                            # HERE need change
                            next_id_frame_case = case + "_" + str(int(fi)+30) + "_" + str(id)
                            print("======", id_frame_case, idx_per_id[id_frame_case])
                            token = dict_by_id[id_frame_case]["annotation"][0]
                            sample_token = sample_dict[fi]
                            instance_token = dict_by_id[id_frame_case]["token"]
                            if idx < TwoDBBOXES_SIZE:
                                pixelRate = TwoDBBOXES[idx]["pixelRate"]
                            else:
                                pixelRate = TwoDBBOXESCULLED[idx - TwoDBBOXES_SIZE]["pixelRate"]
                            visibility_token = str(determine_grade(pixelRate))
                            vel = item["vel"]
                            attribute_tokens = determine_attribute_status(type, vel)
                            translation = pos
                            size = size
                            rotation = utils.rot2fourvar(rot)
                            prev = dict_by_id[pre_id_frame_case]["annotation"][0] if pre_id_frame_case in dict_by_id  else ""
                            next = dict_by_id[next_id_frame_case]["annotation"][0] if next_id_frame_case in dict_by_id else ""
                            num_lidar_pts = 0
                            num_radar_pts = 0
                            has_token_before = False

                            for ann in sample_annotation_json_lst:
                                if token == ann["token"]:
                                    has_token_before = True
                                    break
                            if has_token_before:
                                continue

                            annotation_json = get_sample_annotation(token, sample_token, instance_token,
                                                                    visibility_token,
                                                                    attribute_tokens,
                                                                    translation, size, rotation, prev,
                                                                    next,
                                                                    num_lidar_pts, num_radar_pts)
                            idx_per_id[id_frame_case] += 1
                            sample_annotation_json_lst.append(annotation_json)

            else:

                files = [fi for fi in os.listdir(sensor_path) if fi.endswith('.pcd')]
                files_sorted = sorted(files, key=extract_number)
                stripped_list = [file.rstrip('.pcd') for file in files_sorted]
                for fi in stripped_list:
                    # change area
                    if fi not in sample_dict:
                        continue
                    with open(os.path.join(sensor_path, fi + ".json"), "r") as f:
                        ann = json.load(f)
                        data = ann["bboxes3D"]
                        pos_ego = ann["pos"]

                    for item in data:
                        type = item["type"]
                        if type in utils.simone2nuscenes_lst:
                            pos = [item["pos"][0], item["pos"][1], item["pos"][2]]

                            pos_n = np.array(pos)

                            pos_ego_n = np.array(pos_ego)
                            relpos = pos_n - pos_ego_n
                            if not is_within_range(relpos, utils.point_cloud_range):
                                print("=========================================")
                                continue

                            rot = item["rot"]
                            size = [item["size"][1], item["size"][0], item["size"][2]]
                            id = item["id"]
                            fi = fi.split('.')[0]
                            id_frame_case = case+"_"+fi+"_"+str(id)
                            pre_id_frame_case = ""
                            for fi_tmp in range(int(fi) - 30, -1, -30):
                                candidate = f"{case}_{fi_tmp}_{id}"
                                if candidate in dict_by_id:
                                    pre_id_frame_case = candidate
                                    break

                            next_id_frame_case = ""
                            for fi_tmp in range(int(fi) + 30, 2000, 30):
                                candidate = f"{case}_{fi_tmp}_{id}"
                                if candidate in dict_by_id:
                                    next_id_frame_case = candidate
                                    break

                            token = dict_by_id[id_frame_case]["annotation"][0]
                            sample_token = sample_dict[fi]
                            instance_token = dict_by_id[id_frame_case]["token"]
                            visibility_token = 1
                            visibility_token = str(determine_grade(pixelRate))
                            vel = item["vel"]
                            attribute_tokens = determine_attribute_status(type, vel)
                            translation = pos
                            has_token_before = False
                            for ann in sample_annotation_json_lst:
                                if token == ann["token"]:
                                    # num_lidar_pts = 100
                                    num_lidar_pts = item["totalPoints"]
                                    ann["num_lidar_pts"] = num_lidar_pts
                                    has_token_before = True
                                    break
                            if not has_token_before:
                                size = size
                                rotation = utils.rot2fourvar(rot)
                                prev = dict_by_id[pre_id_frame_case]["annotation"][0] if pre_id_frame_case in dict_by_id else ""
                                next = dict_by_id[next_id_frame_case]["annotation"][0] if next_id_frame_case in dict_by_id else ""
                                num_lidar_pts = item["totalPoints"]
                                num_radar_pts = 0
                                if num_lidar_pts <= 0:
                                    num_lidar_pts = 1

                                annotation_json = get_sample_annotation(token, sample_token, instance_token,
                                                                        visibility_token,
                                                                        attribute_tokens, translation, size, rotation,
                                                                        prev,
                                                                        next,
                                                                        num_lidar_pts, num_radar_pts)
                                idx_per_id[id_frame_case] += 1
                                sample_annotation_json_lst.append(annotation_json)

    sample_annotation_filepath = 'nuscenes_v1.0/v1.0-mini/sample_annotation.json'
    # sample_annotation_filepath = 'sample_annotation1.json'
    with open(sample_annotation_filepath, "w") as f:
        json.dump(sample_annotation_json_lst, f, indent=4)


if __name__ == '__main__':
    main()
