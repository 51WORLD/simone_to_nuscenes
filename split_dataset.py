import utils
import os
import shutil

mapping_dict = {
    'front': 'CAM_FRONT',
    'front_left': 'CAM_FRONT_LEFT',
    'front_right': 'CAM_FRONT_RIGHT',
    'rear': 'CAM_BACK',
    'rear_left': 'CAM_BACK_LEFT',
    'rear_right': 'CAM_BACK_RIGHT',
    'lidar_top': 'LIDAR_TOP'
}

dst_sweeps_path = os.path.join(utils.base_path, 'sweeps')
for fi in list(mapping_dict.values()):
    if not os.path.exists(os.path.join(dst_sweeps_path, fi)):
        os.makedirs(os.path.join(dst_sweeps_path, fi))

train_ratio = utils.TRAIN_RATIO
val_ratio = utils.VAL_RATIO
src_path = [os.path.join(utils.base_path, 'samples', item) for item in mapping_dict.values()]
print(src_path)
for sensor_filepath in src_path:
    if len(os.listdir(sensor_filepath)) < utils.NUM_SAMPLE:
        continue
    id_by_sample = 0
    for fi in sorted(os.listdir(sensor_filepath), key=lambda x: int(x.split(".")[0])):
        if id_by_sample % (train_ratio + val_ratio) != 0:
            src_file = os.path.join(sensor_filepath, fi)
            dest_file = src_file.replace('samples', 'sweeps')
            # Create destination directory if it doesn't exist
            os.makedirs(os.path.dirname(dest_file), exist_ok=True)
            # Check if destination file already exists and remove it
            if os.path.exists(dest_file):
                os.remove(dest_file)
            # Move the file
            shutil.move(src_file, dest_file)
        id_by_sample += 1
