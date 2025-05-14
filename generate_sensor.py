import json
import uuid
import utils

mapping_dict = {
    'front': 'CAM_FRONT',
    'front_left': 'CAM_FRONT_LEFT',
    'front_right': 'CAM_FRONT_RIGHT',
    'rear': 'CAM_BACK',
    'rear_left': 'CAM_BACK_LEFT',
    'rear_right': 'CAM_BACK_RIGHT',
    'lidar_top': 'LIDAR_TOP'
}

def main():
    # Src json
    token_filepath = 'data/token.json'
    with open(token_filepath, "r") as f:
        sensor_data = json.load(f)["sensor"]

    sensor_json_lst = []
    for idx, item in enumerate(mapping_dict.keys()):
        sensor_json = {
            "token": sensor_data[idx]["token"],
            "channel": mapping_dict[item],
            "modality": "camera" if 'lidar' not in item else 'lidar'
        }
        sensor_json_lst.append(sensor_json)
    # Dest json
    sensor_filepath = 'nuscenes_v1.0/v1.0-mini/sensor.json'
    with open(sensor_filepath, 'w') as json_file:
        json.dump(sensor_json_lst, json_file, indent=4)

if __name__ == '__main__':
    main()