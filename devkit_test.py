import os
from nuscenes.nuscenes import NuScenes

import utils

frames_path = 'frames'
if not os.path.exists(frames_path):
    os.makedirs(frames_path)

# Initialize the NuScenes dataset
nusc = NuScenes(version='v1.0-mini', dataroot='nuscenes_v1.0', verbose=True)

for index in range(0, utils.NUM_SCENE):
    # Get the first scene
    my_scene = nusc.scene[index]
    print(f'Processing scene {index} of {utils.NUM_SCENE}')
    print(f'Scene name: {my_scene["name"]}')

    # Get the first sample token of the scene
    sample_token = my_scene['first_sample_token']
    scene_name = my_scene['name']
    # Generate frames of cam + lidar
    id = 0
    # # Loop through all samples in the scene
    while sample_token:
        # Get the sample
        sample = nusc.get('sample', sample_token)

        # Render the sample
        dir = f'frames/{scene_name}'
        if not os.path.exists(dir):
            os.makedirs(dir)
        nusc.render_sample(sample_token, out_path=f'{dir}/{str(id)}_{str(scene_name)}_{sample_token}.png', verbose=False)

        # Move to the next sample
        sample_token = sample['next']
        id += 1
