import json
import os
from uuid import uuid4
import utils
import numpy as np
from uuid import uuid4
import copy
import re
import yaml
from collections import defaultdict


def load_category():
    category_filepath = 'data/category.json'
    with open(category_filepath, "r") as f:
        category_data = json.load(f)
    category_dict = {data["name"]: data["token"] for data in category_data}
    return category_dict


def get_instances(token, category_token, nbr_annotations, first_annotation_token, last_annotation_token):
    instance = {
        "token": token,
        "category_token": category_token,
        "nbr_annotations": nbr_annotations,
        "first_annotation_token": first_annotation_token,
        "last_annotation_token": last_annotation_token
    }
    return instance


def load_simone_to_nuscenes_map():
    with open("simone.yaml", "r") as file:
        data = yaml.safe_load(file)

    simeone2nuscenes_yaml = data["simone_to_nuscenes_map"]
    simone_category = data["simeone_category"]
    # Note: In simone, id including [4,6,8,18,19,21] can be labelled.
    simone2nuscenes_map = {k: v[0] for k, v in simeone2nuscenes_yaml.items() if v!= None}
    simone_category = {k: v for k, v in simone_category.items()}
    return simone2nuscenes_map, simone_category


def generate_sequences(input_file, output_folder):
    with open(input_file, 'r') as f:
        data = json.load(f)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    sequences = defaultdict(list)
    token_dict = {item['token']: item for item in data}

    for item in data:
        instance_token = item['instance_token']
        current_item = item
        while current_item:
            sequences[instance_token].append(current_item)
            if current_item['next'] == " ":
                break
            current_item = token_dict.get(current_item['next'])

    # for instance_token, sequence in sequences.items():
    #     output_file = os.path.join(output_folder, f'{instance_token}_sequence.json')
    #     with open(output_file, 'w') as out_f:
    #         json.dump(sequence, out_f, indent=4)

    return sequences


def main():
    instance_lst = []
    category_dict = load_category()
    simone2nuscenes_map, simone_category = load_simone_to_nuscenes_map()

    input_file = 'nuscenes_v1.0/v1.0-mini/sample_annotation.json'
    output_folder = 'sequence'

    # 生成序列
    sequences = generate_sequences(input_file, output_folder)
    # print(sequences)

    type_filepath = 'data/instances.json'
    with open(type_filepath, "r") as f:
        type_data = json.load(f)

    simone2nuscenes_set = set(utils.simone2nuscenes_lst)
    for instance_token, sequence in sequences.items():
        for key, value in type_data.items():
            if value['token'] == instance_token:
                type = value["type"]
                print(type)
                break

        if type in simone2nuscenes_set:
            simone_typename = simone_category[type]
            nuscene_typename = simone2nuscenes_map[simone_typename]
            category_token = category_dict[nuscene_typename]
            nbr_annotations = len(sequence)
            first_annotation_token = sequence[0]['token']
            last_annotation_token = sequence[-1]['token']
            instance_json = get_instances(instance_token, category_token, nbr_annotations, first_annotation_token,
                                          last_annotation_token)
            instance_lst.append(instance_json)

    instance_filepath = 'nuscenes_v1.0/v1.0-mini/instance.json'
    with open(instance_filepath, "w") as f:
        json.dump(instance_lst, f, indent=4)


if __name__ == '__main__':
    main()
    