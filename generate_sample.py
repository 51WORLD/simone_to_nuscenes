import json
import utils
def generate_sample_json():
    # Src json
    token_filepath = 'data/token.json'
    with open(token_filepath, "r") as f:
        token_data = json.load(f)

    scene_lst = token_data["scene"]
    sample_json_lst = []
    sample_filename_dict = {}
    # for scene_item in scene_lst:
    for index, scene_item in enumerate(scene_lst):
        timestart = utils.timestart + index * 5 * 60 * 1000000
        scene_token = scene_item["token"]
        sample_lst = [si["token"] for si in scene_item["samples"]]
        timestamp = [si["samplename"] for si in scene_item["samples"]]
        print("********************")
        print("timestamp: ", timestamp)
        for idx, item in enumerate(sample_lst):
            data = {
                "token": item,
                "timestamp": int(timestart + int(timestamp[idx]) * 1 / 60 * 1000 * 1000),
                "prev": sample_lst[idx - 1] if idx != 0 else "",
                "next": sample_lst[idx + 1] if idx != len(sample_lst) - 1 else "",
                "scene_token": scene_token
            }
            sample_filename_dict.update({int(timestart + int(timestamp[idx]) * 1 / 60 * 1000 * 1000): item})
            sample_json_lst.append(data)
    return sample_filename_dict, sample_json_lst
def main():
        # Dest json
        _,sample_json_lst=generate_sample_json()
        sample_filepath = 'nuscenes_v1.0/v1.0-mini/sample.json'
        with open(sample_filepath, 'w') as f:
            json.dump(sample_json_lst, f, indent=4, ensure_ascii=False)



if __name__ == '__main__':
    main()
