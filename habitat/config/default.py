#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

import yacs.config

# from habitat.config import Config as CN # type: ignore

# Default Habitat config node
class Config(yacs.config.CfgNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, new_allowed=True)


CN = Config

DEFAULT_CONFIG_DIR = "configs/"
CONFIG_FILE_SEPARATOR = ","

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
_C.SEED = 100
_C.SEED = 42
# -----------------------------------------------------------------------------
# ENVIRONMENT
# -----------------------------------------------------------------------------
_C.ENVIRONMENT = CN()
_C.ENVIRONMENT.MAX_EPISODE_STEPS = 1000
_C.ENVIRONMENT.MAX_EPISODE_SECONDS = 10000000
_C.ENVIRONMENT.ITERATOR_OPTIONS = CN()
_C.ENVIRONMENT.ITERATOR_OPTIONS.CYCLE = True
_C.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = True
_C.ENVIRONMENT.ITERATOR_OPTIONS.GROUP_BY_SCENE = True
_C.ENVIRONMENT.ITERATOR_OPTIONS.NUM_EPISODE_SAMPLE = -1
_C.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = -1
_C.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = int(1e4)
_C.ENVIRONMENT.ITERATOR_OPTIONS.STEP_REPETITION_RANGE = 0.2
# -----------------------------------------------------------------------------
# TASK
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# # NAVIGATION TASK
# -----------------------------------------------------------------------------
_C.TASK = CN()
_C.TASK.TYPE = "Nav-v0"
_C.TASK.SUCCESS_DISTANCE = 0.2
_C.TASK.SENSORS = []
_C.TASK.MEASUREMENTS = []
_C.TASK.AREA_REWARD = ""
#_C.TASK.GOAL_SENSOR_UUID = "multiobjectgoal"
_C.TASK.POSSIBLE_ACTIONS = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
# -----------------------------------------------------------------------------
# # ACTIONS
# -----------------------------------------------------------------------------
ACTIONS = CN()
ACTIONS.STOP = CN()
ACTIONS.STOP.TYPE = "StopAction"
# -----------------------------------------------------------------------------
# # NAVIGATION ACTIONS
# -----------------------------------------------------------------------------
ACTIONS.MOVE_FORWARD = CN()
ACTIONS.MOVE_FORWARD.TYPE = "MoveForwardAction"
ACTIONS.TURN_LEFT = CN()
ACTIONS.TURN_LEFT.TYPE = "TurnLeftAction"
ACTIONS.TURN_RIGHT = CN()
ACTIONS.TURN_RIGHT.TYPE = "TurnRightAction"
ACTIONS.LOOK_UP = CN()
ACTIONS.LOOK_UP.TYPE = "LookUpAction"
ACTIONS.LOOK_DOWN = CN()
ACTIONS.LOOK_DOWN.TYPE = "LookDownAction"
ACTIONS.FOUND = CN()
ACTIONS.FOUND.TYPE = "FoundObjectAction"
ACTIONS.TAKE_PICTURE = CN()
ACTIONS.TAKE_PICTURE.TYPE = "TakePicture"


ACTIONS.TELEPORT = CN()
ACTIONS.TELEPORT.TYPE = "TeleportAction"

_C.TASK.ACTIONS = ACTIONS
# -----------------------------------------------------------------------------
# # TASK SENSORS
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# POINTGOAL SENSOR
# -----------------------------------------------------------------------------
_C.TASK.POINTGOAL_SENSOR = CN()
_C.TASK.POINTGOAL_SENSOR.TYPE = "PointGoalSensor"
_C.TASK.POINTGOAL_SENSOR.GOAL_FORMAT = "POLAR"
_C.TASK.POINTGOAL_SENSOR.DIMENSIONALITY = 2
# -----------------------------------------------------------------------------
# POINTGOAL WITH GPS+COMPASS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR = _C.TASK.POINTGOAL_SENSOR.clone()
_C.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.TYPE = (
    "PointGoalWithGPSCompassSensor"
)
# -----------------------------------------------------------------------------
# OBJECTGOAL SENSOR
# -----------------------------------------------------------------------------
_C.TASK.OBJECTGOAL_SENSOR = CN()
_C.TASK.OBJECTGOAL_SENSOR.TYPE = "ObjectGoalSensor"
_C.TASK.OBJECTGOAL_SENSOR.GOAL_SPEC = "TASK_CATEGORY_ID"
_C.TASK.OBJECTGOAL_SENSOR.GOAL_SPEC_MAX_VAL = 50
# -----------------------------------------------------------------------------
# MULTIOBJECTGOAL SENSOR
# -----------------------------------------------------------------------
_C.TASK.MULTI_GOAL_SENSOR = CN()
_C.TASK.MULTI_GOAL_SENSOR.TYPE = "MultiGoalSensor"
_C.TASK.MULTI_GOAL_SENSOR.GOAL_SPEC = "TASK_CATEGORY_ID"
_C.TASK.MULTI_GOAL_SENSOR.GOAL_SPEC_MAX_VAL = 50
# -----------------------------------------------------------------------------
# POSITION SENSOR
# -----------------------------------------------------------------------------
_C.TASK.POSITION_SENSOR = CN()
_C.TASK.POSITION_SENSOR.TYPE = "PositionSensor"
# -----------------------------------------------------------------------------
# HEADING SENSOR
# -----------------------------------------------------------------------------
_C.TASK.HEADING_SENSOR = CN()
_C.TASK.HEADING_SENSOR.TYPE = "HeadingSensor"
# -----------------------------------------------------------------------------
# COMPASS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.COMPASS_SENSOR = CN()
_C.TASK.COMPASS_SENSOR.TYPE = "CompassSensor"
# -----------------------------------------------------------------------------
# GPS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.GPS_SENSOR = CN()
_C.TASK.GPS_SENSOR.TYPE = "GPSSensor"
_C.TASK.GPS_SENSOR.DIMENSIONALITY = 2
# -----------------------------------------------------------------------------
# PROXIMITY SENSOR
# -----------------------------------------------------------------------------
_C.TASK.PROXIMITY_SENSOR = CN()
_C.TASK.PROXIMITY_SENSOR.TYPE = "ProximitySensor"
_C.TASK.PROXIMITY_SENSOR.MAX_DETECTION_RADIUS = 2.0
# -----------------------------------------------------------------------------
# POSE_ESTIMATION_RGB_SENSOR
# -----------------------------------------------------------------------------
_C.TASK.POSE_ESTIMATION_RGB_SENSOR = CN()
_C.TASK.POSE_ESTIMATION_RGB_SENSOR.TYPE = "PoseEstimationRGBSensor"
_C.TASK.POSE_ESTIMATION_RGB_SENSOR.HEIGHT = 480
_C.TASK.POSE_ESTIMATION_RGB_SENSOR.WIDTH = 640
# -----------------------------------------------------------------------------
# POSE_ESTIMATION_MASK_SENSOR
# -----------------------------------------------------------------------------
_C.TASK.POSE_ESTIMATION_MASK_SENSOR = CN()
_C.TASK.POSE_ESTIMATION_MASK_SENSOR.TYPE = "PoseEstiomationMaskSensor"
# -----------------------------------------------------------------------------
# DELTA SENSOR
# -----------------------------------------------------------------------------
_C.TASK.DELTA_SENSOR = CN()
_C.TASK.DELTA_SENSOR.TYPE = "DeltaSensor"
# -----------------------------------------------------------------------------
# PICTURE MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.PICTURE = CN()
_C.TASK.PICTURE.TYPE = "Picture"
# -----------------------------------------------------------------------------
# CI MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.CI = CN()
_C.TASK.CI.TYPE = "CI"
_C.TASK.CI.LOW_REGION_WEIGHT = 1.0
_C.TASK.CI.MID_REGION_WEIGHT = 3.0
_C.TASK.CI.HIGH_REGION_WEIGHT = 5.0
_C.TASK.CI.LOW_CATEGORY_VALUE = -0.01
_C.TASK.CI.MID_CATEGORY_VALUE = 1.0
_C.TASK.CI.HIGH_CATEGORY_VALUE = 2.0
# -----------------------------------------------------------------------------
# Saliency MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.SALIENCY = CN()
_C.TASK.SALIENCY.TYPE = "Saliency"
# -----------------------------------------------------------------------------
# SUCCESS MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.SUCCESS = CN()
_C.TASK.SUCCESS.TYPE = "Success"
_C.TASK.SUCCESS.SUCCESS_DISTANCE = 0.2
# -----------------------------------------------------------------------------
# SPL MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.SPL = CN()
_C.TASK.SPL.TYPE = "SPL"
# -----------------------------------------------------------------------------
# MSPL MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.MSPL = CN()
_C.TASK.MSPL.TYPE = "MSPL"
# -----------------------------------------------------------------------------
# PSPL MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.PSPL = CN()
_C.TASK.PSPL.TYPE = "PSPL"
# -----------------------------------------------------------------------------
# FOW MAP
# -----------------------------------------------------------------------------
_C.TASK.FOW_MAP = CN()
_C.TASK.FOW_MAP.TYPE = "FowMap"
_C.TASK.FOW_MAP.VISIBILITY_DIST = 6.0
_C.TASK.FOW_MAP.FOV = 79
# -----------------------------------------------------------------------------
# PICTURE_MAP
# -----------------------------------------------------------------------------
_C.TASK.PICTURE_MAP = CN()
_C.TASK.PICTURE_MAP.VISIBILITY_DIST = 6.0
_C.TASK.PICTURE_MAP.FOV = 79
# -----------------------------------------------------------------------------
# TopDownMap MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.TOP_DOWN_MAP = CN()
_C.TASK.TOP_DOWN_MAP.TYPE = "TopDownMap"
_C.TASK.TOP_DOWN_MAP.MAX_EPISODE_STEPS = _C.ENVIRONMENT.MAX_EPISODE_STEPS
_C.TASK.TOP_DOWN_MAP.MAP_PADDING = 3
_C.TASK.TOP_DOWN_MAP.NUM_TOPDOWN_MAP_SAMPLE_POINTS = 20000
_C.TASK.TOP_DOWN_MAP.MAP_RESOLUTION = 1250
_C.TASK.TOP_DOWN_MAP.DRAW_SOURCE = True
_C.TASK.TOP_DOWN_MAP.DRAW_BORDER = True
_C.TASK.TOP_DOWN_MAP.DRAW_SHORTEST_PATH = True
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR = CN()
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR.DRAW = True
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR.VISIBILITY_DIST = 5.0
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR.FOV = 79
_C.TASK.TOP_DOWN_MAP.DRAW_VIEW_POINTS = True
_C.TASK.TOP_DOWN_MAP.DRAW_GOAL_POSITIONS = True
# Axes aligned bounding boxes
_C.TASK.TOP_DOWN_MAP.DRAW_GOAL_AABBS = True
# -----------------------------------------------------------------------------
# NewTopDownMap MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.NEW_TOP_DOWN_MAP = CN()
_C.TASK.NEW_TOP_DOWN_MAP.TYPE = "NewTopDownMap"
_C.TASK.NEW_TOP_DOWN_MAP.MAX_EPISODE_STEPS = _C.ENVIRONMENT.MAX_EPISODE_STEPS
_C.TASK.NEW_TOP_DOWN_MAP.MAP_PADDING = 3
_C.TASK.NEW_TOP_DOWN_MAP.NUM_TOPDOWN_MAP_SAMPLE_POINTS = 20000
_C.TASK.NEW_TOP_DOWN_MAP.MAP_RESOLUTION = 1250
_C.TASK.NEW_TOP_DOWN_MAP.DRAW_BORDER = True
# -----------------------------------------------------------------------------
# ExploredMap MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.EXPLORED_MAP = CN()
_C.TASK.EXPLORED_MAP.TYPE = "ExploredMap"
_C.TASK.EXPLORED_MAP.MAX_EPISODE_STEPS = _C.ENVIRONMENT.MAX_EPISODE_STEPS
_C.TASK.EXPLORED_MAP.MAP_PADDING = 3
_C.TASK.EXPLORED_MAP.NUM_TOPDOWN_MAP_SAMPLE_POINTS = 20000
_C.TASK.EXPLORED_MAP.MAP_RESOLUTION = 1250
_C.TASK.EXPLORED_MAP.DRAW_SOURCE = False
_C.TASK.EXPLORED_MAP.DRAW_BORDER = True
_C.TASK.EXPLORED_MAP.DRAW_SHORTEST_PATH = False
_C.TASK.EXPLORED_MAP.FOG_OF_WAR = CN()
_C.TASK.EXPLORED_MAP.FOG_OF_WAR.DRAW = True
_C.TASK.EXPLORED_MAP.FOG_OF_WAR.VISIBILITY_DIST = 5.0
_C.TASK.EXPLORED_MAP.FOG_OF_WAR.FOV = 79
_C.TASK.EXPLORED_MAP.DRAW_VIEW_POINTS = True
_C.TASK.EXPLORED_MAP.DRAW_GOAL_POSITIONS = True
# Axes aligned bounding boxes
_C.TASK.EXPLORED_MAP.DRAW_GOAL_AABBS = True

# -----------------------------------------------------------------------------
# SmoothCoverage MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.SMOOTH_COVERAGE = CN()
_C.TASK.SMOOTH_COVERAGE.TYPE = "SmoothCoverage"
_C.TASK.SMOOTH_COVERAGE.NUM_TOPDOWN_MAP_SAMPLE_POINTS = 20000
_C.TASK.SMOOTH_COVERAGE.MAP_RESOLUTION = 1250
_C.TASK.SMOOTH_COVERAGE.FOG_OF_WAR = CN()
_C.TASK.SMOOTH_COVERAGE.FOG_OF_WAR.DRAW = True
_C.TASK.SMOOTH_COVERAGE.FOG_OF_WAR.VISIBILITY_DIST = 5.0
_C.TASK.SMOOTH_COVERAGE.FOG_OF_WAR.FOV = 79
# -----------------------------------------------------------------------------
# NoveltyValue MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.NOVELTY_VALUE = CN()
_C.TASK.NOVELTY_VALUE.TYPE = "NoveltyValue"
_C.TASK.NOVELTY_VALUE.NUM_TOPDOWN_MAP_SAMPLE_POINTS = 20000
_C.TASK.NOVELTY_VALUE.MAP_RESOLUTION = 1250
# -----------------------------------------------------------------------------
# PictureRangeMap MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.PICTURE_RANGE_MAP = CN()
_C.TASK.PICTURE_RANGE_MAP.TYPE = "PictureRangeMap"
_C.TASK.PICTURE_RANGE_MAP.MAX_EPISODE_STEPS = _C.ENVIRONMENT.MAX_EPISODE_STEPS
_C.TASK.PICTURE_RANGE_MAP.MAP_PADDING = 3
_C.TASK.PICTURE_RANGE_MAP.NUM_TOPDOWN_MAP_SAMPLE_POINTS = 20000
_C.TASK.PICTURE_RANGE_MAP.MAP_RESOLUTION = 1250
_C.TASK.PICTURE_RANGE_MAP.DRAW_SOURCE = True
_C.TASK.PICTURE_RANGE_MAP.DRAW_BORDER = True
_C.TASK.PICTURE_RANGE_MAP.DRAW_SHORTEST_PATH = True
_C.TASK.PICTURE_RANGE_MAP.FOG_OF_WAR = CN()
_C.TASK.PICTURE_RANGE_MAP.FOG_OF_WAR.DRAW = True
_C.TASK.PICTURE_RANGE_MAP.FOG_OF_WAR.VISIBILITY_DIST = 5.0
_C.TASK.PICTURE_RANGE_MAP.FOG_OF_WAR.FOV = 79
_C.TASK.PICTURE_RANGE_MAP.DRAW_VIEW_POINTS = True
_C.TASK.PICTURE_RANGE_MAP.DRAW_GOAL_POSITIONS = True
# Axes aligned bounding boxes
_C.TASK.PICTURE_RANGE_MAP.DRAW_GOAL_AABBS = True

# -----------------------------------------------------------------------------
# COLLISIONS MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.COLLISIONS = CN()
_C.TASK.COLLISIONS.TYPE = "Collisions"
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# # EQA TASK
# -----------------------------------------------------------------------------
_C.TASK.ACTIONS.ANSWER = CN()
_C.TASK.ACTIONS.ANSWER.TYPE = "AnswerAction"
# # EQA TASK QUESTION SENSOR
# -----------------------------------------------------------------------------
_C.TASK.QUESTION_SENSOR = CN()
_C.TASK.QUESTION_SENSOR.TYPE = "QuestionSensor"
# -----------------------------------------------------------------------------
# # EQA TASK CORRECT_ANSWER measure for training
# -----------------------------------------------------------------------------
_C.TASK.CORRECT_ANSWER = CN()
_C.TASK.CORRECT_ANSWER.TYPE = "CorrectAnswer"
# -----------------------------------------------------------------------------
# # EQA TASK ANSWER SENSOR
# -----------------------------------------------------------------------------
_C.TASK.EPISODE_INFO = CN()
_C.TASK.EPISODE_INFO.TYPE = "EpisodeInfo"
# -----------------------------------------------------------------------------
# # VLN TASK INSTRUCTION SENSOR
# -----------------------------------------------------------------------------
_C.TASK.INSTRUCTION_SENSOR = CN()
_C.TASK.INSTRUCTION_SENSOR.TYPE = "InstructionSensor"
_C.TASK.INSTRUCTION_SENSOR_UUID = "instruction"
# -----------------------------------------------------------------------------
# # DISTANCE_TO_GOAL MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.DISTANCE_TO_GOAL = CN()
_C.TASK.DISTANCE_TO_GOAL.TYPE = "DistanceToGoal"
_C.TASK.DISTANCE_TO_GOAL.DISTANCE_TO = "POINT"
# -----------------------------------------------------------------------------
# # DISTANCE_TO_MULTI_GOAL MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.DISTANCE_TO_MULTI_GOAL = CN()
_C.TASK.DISTANCE_TO_MULTI_GOAL.TYPE = "DistanceToMultiGoal"
_C.TASK.DISTANCE_TO_MULTI_GOAL.DISTANCE_TO = "POINT"
# -----------------------------------------------------------------------------
# # DISTANCE_TO_CURRENT_GOAL MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.DISTANCE_TO_CURR_GOAL = CN()
_C.TASK.DISTANCE_TO_CURR_GOAL.TYPE = "DistanceToCurrGoal"
_C.TASK.DISTANCE_TO_CURR_GOAL.DISTANCE_TO = "POINT"
# -----------------------------------------------------------------------------
# # EPISODE_LENGTH MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.EPISODE_LENGTH = CN()
_C.TASK.EPISODE_LENGTH.TYPE = "EpisodeLength"
# -----------------------------------------------------------------------------
# # SUB_SUCCESS MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.SUB_SUCCESS = CN()
_C.TASK.SUB_SUCCESS.TYPE = "SubSuccess"
_C.TASK.SUB_SUCCESS.SUCCESS_DISTANCE = 1.5
# -----------------------------------------------------------------------------
# # PERCENTAGE_SUCCESS MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.PERCENTAGE_SUCCESS = CN()
_C.TASK.PERCENTAGE_SUCCESS.TYPE = "PercentageSuccess"
_C.TASK.PERCENTAGE_SUCCESS.SUCCESS_DISTANCE = 1.5
# -----------------------------------------------------------------------------
# # RATIO MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.RATIO = CN()
_C.TASK.RATIO.TYPE = "Ratio"
_C.TASK.RATIO.DISTANCE_TO = "POINT"

_C.TASK.RAW_METRICS = CN()
_C.TASK.RAW_METRICS.TYPE = "RawMetrics"
# -----------------------------------------------------------------------------
# # ANSWER_ACCURACY MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.ANSWER_ACCURACY = CN()
_C.TASK.ANSWER_ACCURACY.TYPE = "AnswerAccuracy"
# -----------------------------------------------------------------------------
# SIMULATOR
# -----------------------------------------------------------------------------
_C.SIMULATOR = CN()
_C.SIMULATOR.TYPE = "Sim-v0"
_C.SIMULATOR.ACTION_SPACE_CONFIG = "v0"
_C.SIMULATOR.FORWARD_STEP_SIZE = 0.25  # in metres
_C.SIMULATOR.SCENE = (
    "data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
)
_C.SIMULATOR.SEED = _C.SEED
_C.SIMULATOR.TURN_ANGLE = 10  # angle to rotate left or right in degrees
_C.SIMULATOR.TILT_ANGLE = 15  # angle to tilt the camera up or down in degrees
_C.SIMULATOR.DEFAULT_AGENT_ID = 0
# -----------------------------------------------------------------------------
# SIMULATOR SENSORS
# -----------------------------------------------------------------------------
SIMULATOR_SENSOR = CN()
SIMULATOR_SENSOR.HEIGHT = 480
SIMULATOR_SENSOR.WIDTH = 640
SIMULATOR_SENSOR.HFOV = 79  # horizontal field of view in degrees
SIMULATOR_SENSOR.POSITION = [0, 1.25, 0]
SIMULATOR_SENSOR.ORIENTATION = [0.0, 0.0, 0.0]  # Euler's angles
# -----------------------------------------------------------------------------
# RGB SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.RGB_SENSOR = SIMULATOR_SENSOR.clone()
_C.SIMULATOR.RGB_SENSOR.TYPE = "HabitatSimRGBSensor"
# -----------------------------------------------------------------------------
# DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.DEPTH_SENSOR = SIMULATOR_SENSOR.clone()
_C.SIMULATOR.DEPTH_SENSOR.TYPE = "HabitatSimDepthSensor"
_C.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = 0.0
_C.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = 10.0
_C.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = True
# -----------------------------------------------------------------------------
# SEMANTIC SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.SEMANTIC_SENSOR = SIMULATOR_SENSOR.clone()
_C.SIMULATOR.SEMANTIC_SENSOR.TYPE = "HabitatSimSemanticSensor"
# -----------------------------------------------------------------------------
# AGENT
# -----------------------------------------------------------------------------
_C.SIMULATOR.AGENT_0 = CN()
_C.SIMULATOR.AGENT_0.HEIGHT = 1.5
_C.SIMULATOR.AGENT_0.RADIUS = 0.1
_C.SIMULATOR.AGENT_0.MASS = 32.0
_C.SIMULATOR.AGENT_0.LINEAR_ACCELERATION = 20.0
_C.SIMULATOR.AGENT_0.ANGULAR_ACCELERATION = 4 * 3.14
_C.SIMULATOR.AGENT_0.LINEAR_FRICTION = 0.5
_C.SIMULATOR.AGENT_0.ANGULAR_FRICTION = 1.0
_C.SIMULATOR.AGENT_0.COEFFICIENT_OF_RESTITUTION = 0.0
_C.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR"]
_C.SIMULATOR.AGENT_0.IS_SET_START_STATE = False
_C.SIMULATOR.AGENT_0.START_POSITION = [0, 0, 0]
_C.SIMULATOR.AGENT_0.START_ROTATION = [0, 0, 0, 1]
_C.SIMULATOR.AGENTS = ["AGENT_0"]
# -----------------------------------------------------------------------------
# SIMULATOR HABITAT_SIM_V0
# -----------------------------------------------------------------------------
_C.SIMULATOR.HABITAT_SIM_V0 = CN()
_C.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = 0
# Use Habitat-Sim's GPU->GPU copy mode to return rendering results
# in PyTorch tensors.  Requires Habitat-Sim to be built
# with --with-cuda
# This will generally imply sharing CUDA tensors between processes.
# Read here: https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors
# for the caveats that results in
_C.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = False
# Whether or not the agent slides on collisions
_C.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING = True
_C.SIMULATOR.HABITAT_SIM_V0.ENABLE_PHYSICS = True ## change from False
_C.SIMULATOR.HABITAT_SIM_V0.PHYSICS_CONFIG_FILE = (
    #"./data/default.phys_scene_config.json"
    "./data/default.humanoid_config.json"
)
# -----------------------------------------------------------------------------
# PYROBOT
# -----------------------------------------------------------------------------
_C.PYROBOT = CN()
_C.PYROBOT.ROBOTS = ["locobot"]  # types of robots supported
_C.PYROBOT.ROBOT = "locobot"
_C.PYROBOT.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR", "BUMP_SENSOR"]
_C.PYROBOT.BASE_CONTROLLER = "proportional"
_C.PYROBOT.BASE_PLANNER = "none"
# -----------------------------------------------------------------------------
# SENSORS
# -----------------------------------------------------------------------------
PYROBOT_VISUAL_SENSOR = CN()
PYROBOT_VISUAL_SENSOR.HEIGHT = 480
PYROBOT_VISUAL_SENSOR.WIDTH = 640
# -----------------------------------------------------------------------------
# RGB SENSOR
# -----------------------------------------------------------------------------
_C.PYROBOT.RGB_SENSOR = PYROBOT_VISUAL_SENSOR.clone()
_C.PYROBOT.RGB_SENSOR.TYPE = "PyRobotRGBSensor"
_C.PYROBOT.RGB_SENSOR.CENTER_CROP = False
# -----------------------------------------------------------------------------
# DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.PYROBOT.DEPTH_SENSOR = PYROBOT_VISUAL_SENSOR.clone()
_C.PYROBOT.DEPTH_SENSOR.TYPE = "PyRobotDepthSensor"
_C.PYROBOT.DEPTH_SENSOR.MIN_DEPTH = 0.0
_C.PYROBOT.DEPTH_SENSOR.MAX_DEPTH = 5.0
_C.PYROBOT.DEPTH_SENSOR.NORMALIZE_DEPTH = True
_C.PYROBOT.DEPTH_SENSOR.CENTER_CROP = False
# -----------------------------------------------------------------------------
# BUMP SENSOR
# -----------------------------------------------------------------------------
_C.PYROBOT.BUMP_SENSOR = CN()
_C.PYROBOT.BUMP_SENSOR.TYPE = "PyRobotBumpSensor"
# -----------------------------------------------------------------------------
# ACTIONS LOCOBOT
# -----------------------------------------------------------------------------
_C.PYROBOT.LOCOBOT = CN()
_C.PYROBOT.LOCOBOT.ACTIONS = ["BASE_ACTIONS", "CAMERA_ACTIONS"]
_C.PYROBOT.LOCOBOT.BASE_ACTIONS = ["go_to_relative", "go_to_absolute"]
_C.PYROBOT.LOCOBOT.CAMERA_ACTIONS = ["set_pan", "set_tilt", "set_pan_tilt"]
# TODO(akadian): add support for Arm actions
# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.TYPE = "PointNav-v1"
_C.DATASET.SPLIT = "train"
_C.DATASET.SCENES_DIR = "data/scene_datasets"
_C.DATASET.CONTENT_SCENES = ["*"]
_C.DATASET.DATA_PATH = (
    "data/datasets/pointnav/habitat-test-scenes/v1/{split}/{split}.json.gz"
)

# -----------------------------------------------------------------------------


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.
    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """
    config = _C.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    config.freeze()
    return config