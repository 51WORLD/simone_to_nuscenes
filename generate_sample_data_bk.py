import json
import utils
import re
from uuid import uuid4
import os

num_cam = utils.NUM_CAM
num_lidar = utils.NUM_LIDAR
num_radar = utils.NUM_RADAR

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

def load_token_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def get_sample_data(token, sample_token, ego_pose_token, calibrated_sensor_token, timestamp, fileformat, is_key_frame,
                    height, width, filename, prev, next):
    sample_data = {
        "token": token,
        "sample_token": sample_token,
        "ego_pose_token": ego_pose_token,
        "calibrated_sensor_token": calibrated_sensor_token,
        "timestamp": timestamp,
        "fileformat": fileformat,
        "is_key_frame": is_key_frame,
        "height": height,
        "width": width,
        "filename": filename,
        "prev": prev,
        "next": next
    }
    return sample_data


def main():
    ego_pose_data = load_token_data('data/ego_pose.json')
    ego_pose_lst = [ei["info"]["token"] for ei in ego_pose_data]
    ego_pose_filename = [ei["filename"] for ei in ego_pose_data]

    sample_token_data_lst = []
    sample_data_json_lst = []

    calibrated_sensor_filepath = 'data/calibrated_sensor.json'
    with open(calibrated_sensor_filepath, "r") as f:
        data = json.load(f)

    # 则每len(ego_pose_dict)会出现相同sample_token,在一整个sensor周期内会出现相同的calibrated_sensor,应为dump的一阵个文件夹都是calibrated是相同的
    case_root = 'simone_datasets'
    cases = os.listdir(case_root)
    ego_get_idx_counter = 0

    for index, case in zip(range(len(cases)), cases):
        sample = load_token_data('data/token.json')["scene"][index]["samples"]
        sample_name_lst = [si["samplename"] for si in sample]
        sample_lst = [si["token"] for si in sample]
        # ego_get_idx_counter = 0
        for key, value in mapping_dict.items():
            keys_list = list(mapping_dict.keys())
            sensor_id = keys_list.index(key)

            sensor_path = os.path.join(case_root, case, key)
            # 先写入cam的sample_data
            if key != 'lidar_top':
                files = [fi for fi in os.listdir(sensor_path) if fi != 'DumpSettings.json']
                files_sorted = sorted(files, key=lambda x: int(x))
                max_idx_per_sensor = files_sorted.index(sample_name_lst[-1])

                for idx, fi in enumerate(files_sorted):
                    # 如果当前fi存在于sample_name_lst中，直接取对应sample_token，且"is_key_frame"设置为Ture
                    # 对于单个传感器 只处理小于等于最后一个sample帧号的文件，其余丢弃
                    if idx <= max_idx_per_sensor:
                    # if int(fi) <= int(sample_name_lst[-1]):
                        if fi in sample_name_lst:

                            ego_get_idx = idx + ego_get_idx_counter
                            print(ego_get_idx)
                            sample_name = sample_name_lst[(int(fi) - 300) // 30]
                            # print(sample_name)
                            sample_token = sample_lst[(int(fi) - 300) // 30]

                            ego_pose_token = ego_pose_lst[ego_get_idx]
                            token = ego_pose_token
                            calibrated_sensor_token = data[sensor_id]["token"]

                            ego_pose_name = ego_pose_filename[ego_get_idx]
                            parts = ego_pose_name.split('__')
                            file_name_part = parts[-1]
                            number = file_name_part.split('.')[0]
                            timestamp = number

                            fileformat = utils.FORMAT if sensor_id < num_cam else "pcd"
                            is_key_frame = True
                            height = utils.HEIGHT if sensor_id < num_cam else 0
                            width = utils.WIDTH if sensor_id < num_cam else 0

                            filename = f"samples/{value}/{str(ego_pose_name)}"
                            filename = filename if sensor_id < num_cam else filename + '.bin'

                            prev = ego_pose_lst[ego_get_idx - 1] if idx != 0 else ""
                            next = ego_pose_lst[ego_get_idx + 1] if idx != max_idx_per_sensor else ""

                            sample_data = get_sample_data(token, sample_token, ego_pose_token, calibrated_sensor_token,
                                                          int(timestamp),
                                                          fileformat, is_key_frame,
                                                          height, width, filename, prev, next)
                            sample_data_json_lst.append(sample_data)

                        # 如果当前fi不存在于sample_name_lst中，计算归属的sample_token，且"is_key_frame"设置为False
                        elif fi not in sample_name_lst:
                            ego_get_idx = idx + ego_get_idx_counter
                            # print(ego_get_idx)
                            sample_name = sample_name_lst[((int(fi) - 300) // 30) + 1]
                            # print("((int(fi) - 300) // 30) + 1:", ((int(fi) - 300) // 30) + 1)
                            # print(sample_name)
                            sample_token = sample_lst[((int(fi) - 300) // 30) + 1]

                            ego_pose_token = ego_pose_lst[ego_get_idx]
                            token = ego_pose_token
                            calibrated_sensor_token = data[sensor_id]["token"]

                            ego_pose_name = ego_pose_filename[ego_get_idx]
                            parts = ego_pose_name.split('__')
                            file_name_part = parts[-1]
                            number = file_name_part.split('.')[0]
                            timestamp = number

                            fileformat = utils.FORMAT if sensor_id < num_cam else "pcd"
                            is_key_frame = False
                            height = utils.HEIGHT if sensor_id < num_cam else 0
                            width = utils.WIDTH if sensor_id < num_cam else 0

                            filename = f"sweeps/{value}/{str(ego_pose_name)}"
                            filename = filename if sensor_id < num_cam else filename + '.bin'

                            prev = ego_pose_lst[ego_get_idx - 1] if idx != 0 else ""
                            next = ego_pose_lst[ego_get_idx + 1] if idx != max_idx_per_sensor else ""


                            sample_data = get_sample_data(token, sample_token, ego_pose_token, calibrated_sensor_token,
                                                          int(timestamp),
                                                          fileformat, is_key_frame,
                                                          height, width, filename, prev, next)

                            sample_data_json_lst.append(sample_data)
                    else:
                        break
                # 要根据cam实际获取了多少帧填写 25s 300;55s 660
                ego_get_idx_counter += 660

            else:

                files = [fi for fi in os.listdir(sensor_path) if fi.endswith('.pcd')]
                files_sorted = sorted(files, key=extract_number)
                stripped_list = [file.rstrip('.pcd') for file in files_sorted]
                max_idx_per_sensor = stripped_list.index(sample_name_lst[-1])

                for idx, fi in enumerate(files_sorted):
                    # 如果当前fi存在于sample_name_lst中，直接取对应sample_token，且"is_key_frame"设置为Ture
                    # 对于单个传感器 只处理小于等于最后一个sample帧号的文件，其余丢弃
                    if idx <= max_idx_per_sensor:
                    # if int(fi) <= int(sample_name_lst[-1]):
                        fi = fi.rstrip('.pcd')
                        if fi in sample_name_lst:

                            ego_get_idx = idx + ego_get_idx_counter
                            # print(ego_get_idx)
                            sample_name = sample_name_lst[(int(fi) - 300) // 30]
                            # print(sample_name)
                            sample_token = sample_lst[(int(fi) - 300) // 30]

                            ego_pose_token = ego_pose_lst[ego_get_idx]
                            token = ego_pose_token
                            calibrated_sensor_token = data[sensor_id]["token"]

                            ego_pose_name = ego_pose_filename[ego_get_idx]
                            parts = ego_pose_name.split('__')
                            file_name_part = parts[-1]
                            number = file_name_part.split('.')[0]
                            timestamp = number

                            fileformat = utils.FORMAT if sensor_id < num_cam else "pcd"
                            is_key_frame = True
                            height = utils.HEIGHT if sensor_id < num_cam else 0
                            width = utils.WIDTH if sensor_id < num_cam else 0

                            filename = f"samples/{value}/{str(ego_pose_name)}"
                            filename = filename if sensor_id < num_cam else filename + '.bin'

                            prev = ego_pose_lst[ego_get_idx - 1] if idx != 0 else ""
                            next = ego_pose_lst[ego_get_idx + 1] if idx != max_idx_per_sensor else ""

                            sample_data = get_sample_data(token, sample_token, ego_pose_token, calibrated_sensor_token,
                                                          int(timestamp),
                                                          fileformat, is_key_frame,
                                                          height, width, filename, prev, next)
                            sample_data_json_lst.append(sample_data)

                        # 如果当前fi不存在于sample_name_lst中，计算归属的sample_token，且"is_key_frame"设置为False
                        elif fi not in sample_name_lst:
                            ego_get_idx = idx + ego_get_idx_counter
                            # print(ego_get_idx)
                            # sample_name = sample_name_lst[((int(fi) - 300) // 30) + 1]
                            # print("((int(fi) - 300) // 30) + 1:", ((int(fi) - 300) // 30) + 1)
                            # print(sample_name)
                            sample_token = sample_lst[((int(fi) - 300) // 30) + 1]

                            ego_pose_token = ego_pose_lst[ego_get_idx]
                            token = ego_pose_token
                            calibrated_sensor_token = data[sensor_id]["token"]

                            ego_pose_name = ego_pose_filename[ego_get_idx]
                            parts = ego_pose_name.split('__')
                            file_name_part = parts[-1]
                            number = file_name_part.split('.')[0]
                            timestamp = number

                            fileformat = utils.FORMAT if sensor_id < num_cam else "pcd"
                            is_key_frame = False
                            height = utils.HEIGHT if sensor_id < num_cam else 0
                            width = utils.WIDTH if sensor_id < num_cam else 0

                            filename = f"sweeps/{value}/{str(ego_pose_name)}"
                            filename = filename if sensor_id < num_cam else filename + '.bin'

                            prev = ego_pose_lst[ego_get_idx - 1] if idx != 0 else ""
                            next = ego_pose_lst[ego_get_idx + 1] if idx != max_idx_per_sensor else ""

                            sample_data = get_sample_data(token, sample_token, ego_pose_token, calibrated_sensor_token,
                                                          int(timestamp),
                                                          fileformat, is_key_frame,
                                                          height, width, filename, prev, next)
                            sample_data_json_lst.append(sample_data)
                    else:
                        break
                # 要根据lidar实际获取了多少帧填写  25s 500, 55s 1100
                ego_get_idx_counter += 1100
    sample_token_data_filepath = 'nuscenes_v1.0/v1.0-mini/sample_data.json'
    with open(sample_token_data_filepath, "w") as f:
        json.dump(sample_data_json_lst, f, indent=4)

if __name__ == '__main__':
    main()