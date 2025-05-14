import json


def main():
    # Src json
    ego_pose_src_filepath='data/ego_pose.json'
    with open(ego_pose_src_filepath, 'r') as f:
        ego_pose_data = json.load(f)
    ego_pose_lst=[]
    for item in ego_pose_data:
        ego_pose_lst.append(item["info"])

    # Dest json
    ego_pose_dest_path='nuscenes_v1.0/v1.0-mini/ego_pose.json'
    with open(ego_pose_dest_path, 'w') as f:
        json.dump(ego_pose_lst, f, indent=4)

if __name__ == '__main__':
    main()