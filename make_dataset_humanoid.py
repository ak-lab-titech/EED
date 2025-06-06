import os
import random

import numpy as np
from gym import spaces
import gzip
import pathlib

import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image

from habitat_sim.utils.common import d3_40_colors_rgb
from habitat_baselines.config.default import get_config  
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.datasets.maximum_info.maximuminfo_dataset import MaximumInfoDatasetV1
from habitat.datasets.maximum_info.maximuminfo_generator import generate_maximuminfo_episode, generate_maximuminfo_episode2
from habitat.core.env import Env

from habitat.core.logging import logger

def parse_scene_data(file_path):
    scene_dict = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            parts = line.split(',')
            scene_name = parts[0].strip()
            humanoid_count = int(parts[2].strip())

            # Extract object coordinates
            humanoids = []
            data = []
            coord_start_index = 3
            for i in range(humanoid_count):
                x = float(parts[coord_start_index + i * 4].strip())
                z = float(parts[coord_start_index + i * 4 + 1].strip())
                y = float(parts[coord_start_index + i * 4 + 2].strip())
                rad = float(parts[coord_start_index + i * 4 + 3].strip())
                humanoids.append([x,z,y,rad])

            description_index = coord_start_index + humanoid_count * 4
            add_description = parts[description_index]
            
            data = [humanoids, add_description]
            

            # Add to the dictionary
            if scene_name not in scene_dict:
                scene_dict[scene_name] = []

            scene_dict[scene_name].append(data)

    return scene_dict

   
if __name__ == '__main__':
    save_train = True
    save_val = True
    save_test = False
    
    exp_config = "./habitat_baselines/config/maximuminfo/ppo_maximuminfo_humanoid.yaml"
    opts = None
    config = get_config(exp_config, opts)
    #print(config)
        
    config.defrost()
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = 0.0
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = 5.0
    config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT = 1.5
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR", "SEMANTIC_SENSOR"]
    config.TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = 256
    config.TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.WIDTH = 256
    config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.PHYSICS_CONFIG_FILE = ("./data/default.phys_scene_config.json")
    config.TASK_CONFIG.TRAINER_NAME = "oracle-ego"
    config.freeze()
    
    dir_path = "data/scene_datasets/mp3d"
    dirs = [f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]
    humanoid_position_dataset = parse_scene_data("data/humanoid_position.txt")
    logger.info(humanoid_position_dataset)
    
    train_scene_num = 61
    val_scene_num = 11
    test_scene_num = 18
    episode_num = 20000
    
    i = 0
    dataset_path = "data/datasets/maximuminfo/humanoid/"
    dataset_train = MaximumInfoDatasetV1()
    dataset_val = MaximumInfoDatasetV1()
    dataset_test = MaximumInfoDatasetV1()
    split = ""
    #フォルダがない場合は、作成
    p_dir = pathlib.Path(dataset_path + "train")
    if not p_dir.exists():
        p_dir.mkdir(parents=True)
    p_dir = pathlib.Path(dataset_path + "val")
    if not p_dir.exists():
        p_dir.mkdir(parents=True)
    p_dir = pathlib.Path(dataset_path + "test")
    if not p_dir.exists():
        p_dir.mkdir(parents=True)
        
    # ファイルを読み込んで行ごとにリストに格納する
    with open('data/scene_datasets/mp3d/description.txt', 'r') as file:
        lines = file.readlines()

    # scene id と文章を抽出してデータフレームに変換する
    scene_ids = []
    types = []
    descriptions = []
    for i in range(0, len(lines), 3):
        scene_ids.append(lines[i].strip())
        types.append(lines[i+1].strip())
        descriptions.append(lines[i+2].strip())

    df = pd.DataFrame({'scene_id': scene_ids, 'type': types, 'description': descriptions})      
            
    logger.info(dirs)
    i = 0
    while(True):
        logger.info("###################")
        logger.info(f"i:{i}")
        if i >= len(dirs):
            break
        scene = dirs[i]
        scene_type = df[df["scene_id"]==scene]["type"].item()
        logger.info(f"scene: {scene}, type: {scene_type}")
        if df[df["scene_id"]==scene]["type"].item()=="train":
            if save_train == False:
                i += 1
                continue
            config.defrost()
            split = "train"
            episode_num = 200000
            config.TASK_CONFIG.SIMULATOR.SCENE = "data/scene_datasets/mp3d/" + scene + "/" + scene + ".glb"
            config.TASK_CONFIG.DATASET.DATA_PATH = dataset_path + split + "/" + split +  ".json.gz"
            config.freeze()
        
            sim = HabitatSim(config=config.TASK_CONFIG.SIMULATOR)
            dataset_train.episodes += generate_maximuminfo_episode2(sim=sim, num_episodes=episode_num, humanoid_position_dataset=humanoid_position_dataset,scene=scene)
            
        elif df[df["scene_id"]==scene]["type"].item()=="val":
            if save_val == False:
                i += 1
                continue
            config.defrost()
            split = "val"
            episode_num = 10
            config.TASK_CONFIG.SIMULATOR.SCENE = "data/scene_datasets/mp3d/" + scene + "/" + scene + ".glb"
            config.TASK_CONFIG.DATASET.DATA_PATH = dataset_path + split + "/" + split +  ".json.gz"
            config.freeze()
        
            sim = HabitatSim(config=config.TASK_CONFIG.SIMULATOR)
            #dataset_val.episodes += generate_maximuminfo_episode(sim=sim, num_episodes=episode_num, humanoid_position_dataset=humanoid_position_dataset,scene=scene)
            dataset_val.episodes += generate_maximuminfo_episode2(sim=sim, num_episodes=episode_num, humanoid_position_dataset=humanoid_position_dataset,scene=scene)

        elif df[df["scene_id"]==scene]["type"].item()=="test":
            if save_test == False:
                i += 1
                continue
            config.defrost()
            split = "test"
            episode_num = 20
            config.TASK_CONFIG.SIMULATOR.SCENE = "data/scene_datasets/mp3d/" + scene + "/" + scene + ".glb"
            config.TASK_CONFIG.DATASET.DATA_PATH = dataset_path + split + "/" + split +  ".json.gz"
            config.freeze()
        
            sim = HabitatSim(config=config.TASK_CONFIG.SIMULATOR)
            dataset_test.episodes += generate_maximuminfo_episode2(sim=sim, num_episodes=episode_num, humanoid_position_dataset=humanoid_position_dataset,scene=scene)
            
        else:
            break
    
        logger.info(str(i) + ": SPLIT:train, NUM:" + str(episode_num) + ", TOTAL_NUM:" + str(len(dataset_train.episodes)))
        logger.info("SCENE:" + scene)
        sim.close()
        
        i += 1

    #datasetを.gzに圧縮
    if save_train:
        with gzip.open(dataset_path + "train/train.json.gz", "wt") as f:
            random.shuffle(dataset_train.episodes)
            f.write(dataset_train.to_json())
            print("save train")
    #datasetを.gzに圧縮
    if save_val:
        with gzip.open(dataset_path + "val/val.json.gz", "wt") as f:
            f.write(dataset_val.to_json())
            print("save val")
    #datasetを.gzに圧縮
    if save_test:
        with gzip.open(dataset_path + "test/test.json.gz", "wt") as f:
            f.write(dataset_test.to_json())
            print("save test")