import json
import utils

def main():
    with open('data/calibrated_sensor.json', 'r') as f:
        data = json.load(f)
    calibrated_sensor_json_path = 'nuscenes_v1.0/v1.0-mini/calibrated_sensor.json'
    with open(calibrated_sensor_json_path, 'w') as f:
        json.dump(data, f, indent=4)
if __name__ == '__main__':
    main()