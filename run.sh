#!/bin/bash

# Ensure the script exits on any error
set -e

# # Step 1: Preprocess
# echo "Running preprocess.py..."
# python preprocess.py

# # Step 2: Convert RAW to JPG and copy Lidar folder
# echo "Running RAW2JPG.py..."
# python RAW2JPG.py

# Step 3: Pre-generate token and ego_pose
echo "Running create_token.py..."
python create_token.py

# Step 4: Generate scene, sensor, sample, and ego_pose JSON files
echo "Running generate_scene.py..."
python generate_scene.py

echo "Running generate_sample.py..."
python generate_sample.py

echo "Running generate_sensor.py..."
python generate_sensor.py

echo "Running generate_ego_pose.py..."
python generate_ego_pose.py


# Step 5: Generate calibrated.json and sample_data.json
echo "Running create_calibrated_sensor_token.py..."
python create_calibrated_sensor_token.py

echo "Running generate_calibrated_sensor.py..."
python generate_calibrated_sensor.py

echo "Running generate_sample_data.py..."
python generate_sample_data.py

# Step 6: Generate sample_annotation and instances
echo "Running generate_sample_annotation.py..."
python generate_sample_annotation.py

# Step 7: Run nuscene check tools
echo "Running generate_instances.py..."
python generate_instances.py

#echo "Running Bev Checker"
#python devkit_test.py
#echo "All steps completed successfully."
