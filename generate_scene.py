import json


def main():
    # Src json
    token_filepath = 'data/token.json'
    with open(token_filepath, "r") as f:
        token_data = json.load(f)
    log_filepath = "data/log.json"
    with open(log_filepath, "r") as f:
        log_data = json.load(f)
    log_lst = [log["token"] for log in log_data]

    scene_lst = token_data["scene"]
    scene_json_lst = []
    for idx, item in enumerate(scene_lst):
        scene_data = {
            "token": item["token"],
            "nbr_samples": len(item["samples"]),
            "log_token": log_lst[1],
            "first_sample_token": item["samples"][0]["token"],
            "last_sample_token": item["samples"][-1]["token"],
            "name": item["scene-name"],
            "description": "",  ### optional
        }
        scene_json_lst.append(scene_data)

    # Dest json
    scene_filepath = 'nuscenes_v1.0/v1.0-mini/scene.json'
    with open(scene_filepath, 'w') as f:
        json.dump(scene_json_lst, f, indent=4)


if __name__ == '__main__':
    main()
