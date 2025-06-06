#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import numpy as np
import random

import habitat_sim
from habitat.core.simulator import Simulator
from habitat.datasets.utils import get_action_shortest_path
from habitat.tasks.nav.nav import MaximumInformationEpisode, HumanoidData
from habitat.core.logging import logger

r"""A minimum radius of a plane that a point should be part of to be
considered  as a target or source location. Used to filter isolated points
that aren't part of a floor.
"""
ISLAND_RADIUS_LIMIT = 1.5

def _ratio_sample_rate(ratio: float, ratio_threshold: float) -> float:
    r"""Sampling function for aggressive filtering of straight-line
    episodes with shortest path geodesic distance to Euclid distance ratio
    threshold.

    :param ratio: geodesic distance ratio to Euclid distance
    :param ratio_threshold: geodesic shortest path to Euclid
    distance ratio upper limit till aggressive sampling is applied.
    :return: value between 0.008 and 0.144 for ratio [1, 1.1]
    """
    assert ratio < ratio_threshold
    return 20 * (ratio - 0.98) ** 2


def is_compatible_episode(
    s, t, sim, near_dist, far_dist, geodesic_to_euclid_ratio
):
    euclid_dist = np.power(np.power(np.array(s) - np.array(t), 2).sum(0), 0.5)
    if np.abs(s[1] - t[1]) > 0.5:  # check height difference to assure s and
        #  t are from same floor
        return False, 0
    d_separation = sim.geodesic_distance(s, [t])
    if d_separation == np.inf:
        return False, 0
    if not near_dist <= d_separation <= far_dist:
        return False, 0
    distances_ratio = d_separation / euclid_dist
    if distances_ratio < geodesic_to_euclid_ratio and (
        np.random.rand()
        > _ratio_sample_rate(distances_ratio, geodesic_to_euclid_ratio)
    ):
        return False, 0
    if sim.island_radius(s) < ISLAND_RADIUS_LIMIT:
        return False, 0
    return True, d_separation


def _create_episode(
    episode_id, 
    scene_id, 
    start_position, 
    start_rotation,
    humanoid_ids,
    humanoid_positions,
    humanoid_rotations,
    add_description
) -> Optional[MaximumInformationEpisode]:
    humanoids = []
    #logger.info(f"humanoid_ids: {humanoid_ids}")
    #logger.info(f"humanoid_positions: {humanoid_positions}")
    #logger.info(f"humanoid_rotations: {humanoid_rotations}")
    #logger.info(f"add_description: {add_description}")
    for i in range(len(humanoid_ids)):
        humanoid_id = humanoid_ids[i]
        humanoid_pos = humanoid_positions[i]
        humanoid_rot = humanoid_rotations[i]
        humanoid = HumanoidData(humanoid_id, np.array(humanoid_pos), humanoid_rot)
        humanoids.append(humanoid)

    return MaximumInformationEpisode(
        episode_id=str(episode_id),
        scene_id=scene_id,
        start_position=start_position,
        start_rotation=start_rotation,
        humanoids = humanoids,
        add_description=add_description
    )

def cal_dist(source_position, goal_position, step_size=0.25, episode_step=50):
    if np.abs(source_position[1]-goal_position[1]) > 1.0:
        #logger.info(f"z_dist={np.abs(source_position[1]-goal_position[1])}")
        return False
    dist = (source_position[0]-goal_position[0]) * (source_position[0]-goal_position[0])
    dist += (source_position[1]-goal_position[1]) * (source_position[1]-goal_position[1])
    dist += (source_position[2]-goal_position[2]) * (source_position[2]-goal_position[2])
    dist = np.sqrt(dist)
    #logger.info(f"dist={dist}")

    if dist <= step_size*episode_step:
        return True
    return False

def generate_maximuminfo_episode(
    sim: Simulator, 
    num_episodes: int = -1, 
    is_gen_shortest_path: bool = True,
    shortest_path_success_distance: float = 0.2,
    shortest_path_max_steps: int = 500,
    closest_dist_limit: float = 1,
    furthest_dist_limit: float = 30,
    geodesic_to_euclid_min_ratio: float = 1.1,
    number_retries_per_target: int = 10,
    humanoid_position_dataset = None,
    scene: str = None,
) -> MaximumInformationEpisode:
    episode_count = 0
    while episode_count < num_episodes or num_episodes < 0:
        if episode_count % 10000 == 0:
        #if episode_count % 1 == 0:
            logger.info(f"episode_count: {episode_count}")
        target_position = sim.sample_navigable_point()


        if sim.island_radius(target_position) < ISLAND_RADIUS_LIMIT:
            continue
        

        is_compatible = False
        for retry in range(number_retries_per_target):
            source_position = sim.sample_navigable_point()
            #source_position = np.array([4.942, 3.5590622, -2.938])

            is_compatible, dist = is_compatible_episode(
                source_position,
                target_position,
                sim,
                near_dist=closest_dist_limit,
                far_dist=furthest_dist_limit,
                geodesic_to_euclid_ratio=geodesic_to_euclid_min_ratio,
            )

            if is_compatible:
                break

        #source_position = sim.sample_navigable_point()
        #is_compatible = True
        if is_compatible:
            angle = np.random.uniform(0, 2 * np.pi)
            source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]

            scene_id = sim.config.SCENE
            humanoid_data = humanoid_position_dataset[scene]

            position_flag = False
            position_list = []
            for i in range(len(humanoid_data)):
                #logger.info(humanoid_data[i][0])
                ex_humanoid_position = [humanoid_data[i][0][0][0], humanoid_data[i][0][0][1], humanoid_data[i][0][0][2]] 
                if cal_dist(source_position, ex_humanoid_position) == False:
                    #logger.info("cal_dist")
                    continue

                shortest_paths = get_action_shortest_path(
                                    sim,
                                    source_position=source_position,
                                    source_rotation=source_rotation,
                                    goal_position=ex_humanoid_position,
                                    success_distance=shortest_path_success_distance,
                                    max_episode_steps=1000,
                                )
                if shortest_paths is not None:
                    position_flag = True
                    position_list.append(i)
                
            if position_flag == True:
                position_id = random.randrange(len(position_list))
                pos_id = position_list[position_id]
                humanoids_num = len(humanoid_data[pos_id][0])
                #logger.info(f"humanoids_num: {humanoids_num}")

                humanoid_ids = [random.randrange(8) for _ in range(humanoids_num)]
                
                humanoid_positions = []
                humanoid_rotations = []
                for i in range(humanoids_num):
                    human_pos = humanoid_data[pos_id][0][i]
                    humanoid_positions.append([human_pos[0], human_pos[1], human_pos[2]])
                    humanoid_rotations.append(human_pos[3])
                add_description = humanoid_data[pos_id][1]
                
                """
                #humanoid_ids = [random.randrange(8) for _ in range(2)]
                humanoid_ids = [random.randrange(8) for _ in range(1)]
                # z軸はエージェントの+0.75ぐらい?
                #humanoid_positions = [np.array([4.5265017, 4.3, -4.9547048]), np.array([4.2265017, 4.3, -4.7547048])]
                humanoid_positions = [np.array([4.5265017, 4.3, -4.9547048])]
                #humanoid_rotations = [-1.57, 0.0]
                humanoid_rotations = [-1.57]
                humanoid_rotations = [0.0]
                humanoid_rotations = [1.57]
                humanoid_rotations = [3.14]
                """

                episode = _create_episode(
                    episode_id=episode_count,
                    scene_id=scene_id,
                    start_position=source_position,
                    start_rotation=source_rotation,
                    humanoid_ids=humanoid_ids,
                    humanoid_positions=humanoid_positions,
                    humanoid_rotations=humanoid_rotations,
                    add_description=add_description  
                )

                episode_count += 1
                yield episode
            
        
def generate_maximuminfo_episode2(
    sim: Simulator, 
    num_episodes: int = -1, 
    is_gen_shortest_path: bool = True,
    shortest_path_success_distance: float = 0.2,
    shortest_path_max_steps: int = 500,
    closest_dist_limit: float = 1,
    furthest_dist_limit: float = 30,
    geodesic_to_euclid_min_ratio: float = 1.1,
    number_retries_per_target: int = 10,
    humanoid_position_dataset = None,
    scene: str = None,
) -> MaximumInformationEpisode:
    episode_count = 0
    while episode_count < num_episodes or num_episodes < 0:
        if episode_count % 10000 == 0:
        #if episode_count % 1 == 0:
            logger.info(f"episode_count: {episode_count}")
        target_position = sim.sample_navigable_point()

        if sim.island_radius(target_position) < ISLAND_RADIUS_LIMIT:
            continue
        
        
        is_compatible = False
        for retry in range(number_retries_per_target):
            source_position = sim.sample_navigable_point()
            #source_position = np.array([4.942, 3.5590622, -2.938])

            is_compatible, dist = is_compatible_episode(
                source_position,
                target_position,
                sim,
                near_dist=closest_dist_limit,
                far_dist=furthest_dist_limit,
                geodesic_to_euclid_ratio=geodesic_to_euclid_min_ratio,
            )

            if is_compatible:
                break

        source_position = sim.sample_navigable_point()
        #is_compatible = True
        if is_compatible:
            angle = np.random.uniform(0, 2 * np.pi)
            source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]

            scene_id = sim.config.SCENE
            humanoid_data = humanoid_position_dataset[scene]

            position_flag = False
            position_list = []
            for i in range(len(humanoid_data)):
                #logger.info(humanoid_data[i][0])
                ex_humanoid_position = [humanoid_data[i][0][0][0], humanoid_data[i][0][0][1], humanoid_data[i][0][0][2]] 
                if cal_dist(source_position, ex_humanoid_position) == False:
                    #logger.info("cal_dist")
                    continue

                position_flag = True
                position_list.append(i)
                
            if position_flag == True:
                position_id = random.randrange(len(position_list))
                pos_id = position_list[position_id]
                humanoids_num = len(humanoid_data[pos_id][0])
                #logger.info(f"humanoids_num: {humanoids_num}")

                humanoid_ids = [random.randrange(8) for _ in range(humanoids_num)]
                
                humanoid_positions = []
                humanoid_rotations = []
                for i in range(humanoids_num):
                    human_pos = humanoid_data[pos_id][0][i]
                    humanoid_positions.append([human_pos[0], human_pos[1], human_pos[2]])
                    humanoid_rotations.append(human_pos[3])
                add_description = humanoid_data[pos_id][1]

                episode = _create_episode(
                    episode_id=episode_count,
                    scene_id=scene_id,
                    start_position=source_position,
                    start_rotation=source_rotation,
                    humanoid_ids=humanoid_ids,
                    humanoid_positions=humanoid_positions,
                    humanoid_rotations=humanoid_rotations,
                    add_description=add_description  
                )

                episode_count += 1
                yield episode


def generate_maximuminfo_episode3(
    sim: Simulator, 
    num_episodes: int = -1, 
    is_gen_shortest_path: bool = True,
    shortest_path_success_distance: float = 0.2,
    shortest_path_max_steps: int = 500,
    closest_dist_limit: float = 1,
    furthest_dist_limit: float = 30,
    geodesic_to_euclid_min_ratio: float = 1.1,
    number_retries_per_target: int = 10,
    humanoid_position_dataset = None,
    scene: str = None,
) -> MaximumInformationEpisode:
    episode_count = 0
    while episode_count < num_episodes or num_episodes < 0:
        #if episode_count % 10000 == 0:
        if episode_count % 1 == 0:
            logger.info(f"episode_count: {episode_count}")
        target_position = sim.sample_navigable_point()


        if sim.island_radius(target_position) < ISLAND_RADIUS_LIMIT:
            continue
        

        is_compatible = False
        for retry in range(number_retries_per_target):
            #source_position = sim.sample_navigable_point()
            #source_position = np.array([-0.5,0.85,-32.0]) # ur6pFq6Qu1A
            source_position = np.array([4.942, 3.5590622, -2.938]) # TbHJrupSAjP

            is_compatible, dist = is_compatible_episode(
                source_position,
                target_position,
                sim,
                near_dist=closest_dist_limit,
                far_dist=furthest_dist_limit,
                geodesic_to_euclid_ratio=geodesic_to_euclid_min_ratio,
            )

            if is_compatible:
                break

        #source_position = sim.sample_navigable_point()
        is_compatible = True
        if is_compatible:
            angle = np.random.uniform(0, 2 * np.pi)
            source_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]

            scene_id = sim.config.SCENE
        
            #humanoid_ids = [random.randrange(8) for _ in range(2)]
            # 薄紫, 黄, 白, 緑
            humanoid_ids = [0, 1, 2, 3]
            humanoid_ids = [4, 5, 6, 7]
            #humanoid_ids = [0]
            #humanoid_ids = [3]
            # z軸はエージェントの+0.75ぐらい?
            # TbHJrupSAjP
            humanoid_positions = [np.array([4.5265017, 4.3, -4.9547048]), np.array([4.4265017, 4.3, -4.9547048]), np.array([4.5265017, 4.3, -4.8547048]), np.array([4.4265017, 4.3, -4.8547048])]
            humanoid_positions = [
                                    np.array([4.5265017, 4.3, -4.9547048]), 
                                    np.array([5.5265017, 4.3, -4.9547048]), 
                                    np.array([4.5265017, 4.3, -3.9547048]), 
                                    np.array([5.5265017, 4.3, -3.9547048]),
                                ]
            
            # ur6pFq6Qu1A
            #humanoid_positions = [np.array([-0.6,0.85,-33.0])]
            
            humanoid_rotations = [-1.57, 0.0]
            humanoid_rotations = [-1.57]
            humanoid_rotations = [0.0]
            humanoid_rotations = [1.57]
            humanoid_rotations = [3.14]
            humanoid_rotations = [0, 1.57, 3.14, -1.57]
            humanoid_rotations = [0, 0, 0, 0]
            humanoid_rotations = [1.57, 1.57, 1.57, 1.57]
            humanoid_rotations = [3.14, 3.14, 3.14, 3.14]
            humanoid_rotations = [-1.57, -1.57, -1.57, -1.57]


            episode = _create_episode(
                episode_id=episode_count,
                scene_id=scene_id,
                start_position=source_position,
                start_rotation=source_rotation,
                humanoid_ids=humanoid_ids,
                humanoid_positions=humanoid_positions,
                humanoid_rotations=humanoid_rotations,
                add_description=""  
            )

            episode_count += 1
            yield episode