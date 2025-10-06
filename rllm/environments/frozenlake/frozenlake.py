# DISCLAIMER:
# This implementation is based on the Gymnasium FrozenLake environment and the RAGEN project:
# - Gymnasium: https://gymnasium.farama.org/environments/toy_text/frozen_lake/
# - RAGEN: https://github.com/RAGEN-AI/RAGEN/blob/main/ragen/env/frozen_lake/env.py
#
# Some components have been modified or extended for custom use in this project.

import os
# Set pygame to use headless rendering (must be before pygame/gymnasium import)
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import copy
import logging
import base64
import io
from PIL import Image

import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv as GymFrozenLakeEnv
from gymnasium.utils import seeding

from rllm.environments.base.base_env import BaseEnv

logger = logging.getLogger(__name__)

MAX_STEPS: int = 5


# DFS to check that it's a valid path.
def is_valid(board: list[list[str]], max_size: int) -> bool:
    frontier, discovered = [], set()
    # find the start point
    start_r, start_c = np.where(np.array(board) == "S")
    frontier.append((start_r[0], start_c[0], 0))  # row, col steps
    # dfs to check if there is a path from start to goal
    while frontier:
        r, c, steps = frontier.pop()
        if steps > MAX_STEPS:
            continue

        if (r, c) not in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                    continue
                if board[r_new][c_new] == "G":
                    return True
                if board[r_new][c_new] != "H":
                    frontier.append((r_new, c_new, steps + 1))
    return False


def generate_random_map(size: int = 8, p: float = 0.8, seed: int = 0) -> tuple[list[str], tuple[int, int]]:
    """Generates a random valid map (one that has a path from start to goal)

    Args:
        size: size of each side of the grid
        p: probability that a tile is frozen
        seed: seed to ensure the generation of reproducible maps

    Returns:
        A random valid map
    """
    valid = False
    board: list[list[str]] = []  # initialize to make pyright happy

    np_random, _ = seeding.np_random(seed)

    # generate random start and end points

    while not valid:
        p = min(1, p)
        board = np_random.choice(["F", "H"], (size, size), p=[p, 1 - p]).tolist()

        while True:
            start_r = int(np_random.integers(0, size))
            start_c = int(np_random.integers(0, size))
            goal_r = int(np_random.integers(0, size))
            goal_c = int(np_random.integers(0, size))

            # Ensure start and goal are different positions
            if (start_r, start_c) != (goal_r, goal_c):
                break

        board[start_r][start_c] = "S"
        board[goal_r][goal_c] = "G"

        valid = is_valid(board, size)
    return ["".join(x) for x in board], (goal_r, goal_c)


def get_goal_position(random_map):
    positions = np.argwhere(random_map == b"G")
    if positions.size == 0:
        return None  # G not found
    return tuple(positions[0])  # returns (row, col)



def encode_image(image_content):
    """Encode a numpy array image to base64 PNG string."""
    img = Image.fromarray(image_content)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")   



class FrozenLakeEnv(GymFrozenLakeEnv, BaseEnv):
    """
    Inherits from gymnasium.envs.toy_text.frozen_lake.FrozenLakeEnv

    ## Description
    The game starts with the player at random location of the frozen lake grid world with the
    goal located at another random location for the 4x4 environment.

    ## Action Space
    The action shape is `(1,)` in the range `{0, 3}` indicating
    which direction to move the player.
    NOTE the action space is different from gymnasium.envs.toy_text.frozen_lake.FrozenLakeEnv, start from 1
    - 0: Still
    - 1: Left
    - 2: Down
    - 3: Right
    - 4: Up

    ## Starting State
    The episode starts with the player at random location

    ## Rewards
    NOTE added -0.1 as penalty for invalid action
    Reward schedule:
    - Reach goal: +1
    - Reach hole: 0
    - Reach frozen: 0

    ## Arguments
    `is_slippery`: if action is left and is_slippery is True, then:
    - P(move left)=1/3
    - P(move up)=1/3
    - P(move down)=1/3

    ## Example
    P   _   _   _
    _   _   _   O
    O   _   O   _
    O   _   _   G
    """

    # Map gym state in integer
    MAP_LOOKUP = {
        b"P": 0,
        b"F": 1,
        b"H": 2,
        b"G": 3,
    }

    # Define rules to transform to rendered text observation of the environment
    GRID_LOOKUP = {
        0: " P \t",  # player
        1: " _ \t",  # frozen
        2: " O \t",  # hole
        3: " G \t",  # goal
        4: " X \t",  # player fall into hole
        5: " √ \t",  # player on goal
    }

    ACTION_LOOKUP = {
        0: "None",
        1: "Left",
        2: "Down",
        3: "Right",
        4: "Up",
    }

    INVALID_ACTION = 0
    PENALTY_FOR_INVALID = -1

    def __init__(self, **kwargs):
        global MAX_STEPS
        MAX_STEPS = kwargs.pop("max_steps", 5)

        desc = kwargs.pop("desc", None)
        is_slippery = kwargs.pop("is_slippery", False)
        size = kwargs.pop("size", 8)
        p = kwargs.pop("p", 0.8)
        seed = kwargs.pop("seed", 42)
        self.seed = seed
        self.size = size
        self.p = p

        if desc is None:
            random_map, goal_position = generate_random_map(size=size, p=p, seed=seed)
        else:
            random_map = np.asarray(copy.deepcopy(desc), dtype="c")
            goal_position = get_goal_position(random_map)

        self.goal_postion = goal_position

        GymFrozenLakeEnv.__init__(self, desc=random_map[:], is_slippery=is_slippery)
        self.ACTION_SPACE = gym.spaces.Discrete(4, start=1)

        self.map_kwargs = {
            "size": size,
            "p": p,
        }
        self.env_kwargs = {
            "is_slippery": is_slippery,
            "desc": copy.deepcopy(desc),
            "seed": seed,
        }
        self.action_map = {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
        }  # map from custom Env action to action defined in FrozenLakeEnv in gymnasium

        self.reward = 0
        self._valid_actions = []
        self.step_count = 0

    def _get_player_position(self):
        return (self.s // self.ncol, self.s % self.ncol)  # (row, col)

    def reset(self):
        self.__init__(size=self.map_kwargs["size"], p=self.map_kwargs["p"], seed=self.seed, is_slippery=self.env_kwargs["is_slippery"], desc=self.desc)
        GymFrozenLakeEnv.reset(self, seed=self.seed)
        # obs = self.render(mode="tiny_rgb_array")
        obs = self.render(mode="encoded_image")

        logger.info("🔄" * 30)
        logger.info("ENVIRONMENT RESET - NEW EPISODE")
        logger.info("🔄" * 30)
        # logger.info(f"Initial Grid State:")
        # logger.info(f"\n{obs}")
        logger.info(f"Grid Size: {self.size}x{self.size}")
        logger.info(f"Goal Position: {self.goal_postion}")
        logger.info(f"Is Slippery: {self.env_kwargs['is_slippery']}")
        logger.info("🔄" * 30)
        
        return obs, {}

    def finished(self):
        player_pos = self._get_player_position()
        return self.desc[player_pos] in b"GH"

    def success(self):
        """
        Check if the agent has reacched the goal (G) or hole (H)
        """
        player_pos = self._get_player_position()
        return self.desc[player_pos] in b"G"

    def step(self, action: int):
        """
        - Map custom action to gymnasium FrozenLakeEnv action and take the step
        - Check if the action is effective (whether player moves in the env).
        """
        self.step_count += 1
        action_name = self.ACTION_LOOKUP.get(int(action) if action else 0, 'Unknown')
        logger.info("*" * 60)
        logger.info(f"🎮 ENVIRONMENT ACTION: {action} ({action_name})")
        logger.info("*" * 60)
        
        if self.success():
            logger.info("✅ Already at goal, returning success")
            return self.render(), 1, True, {"action_is_effective": False}

        if not action:
            action = self.INVALID_ACTION
        action = int(action)
        assert isinstance(action, int), "Action must be an integer"
        assert not self.success(), "Agent has already reached the goal or hole"

        if action == self.INVALID_ACTION:  # no penalty for invalid action
            logger.warning("❌ INVALID ACTION - No movement will occur")
            return self.render(), 0, False, {"action_is_effective": False}

        prev_player_position = int(self.s)
        prev_player_coords = self._get_player_position()
        logger.info(f"Player position before: {prev_player_coords} (index: {prev_player_position})")

        player_pos, reward, done, _, prob = GymFrozenLakeEnv.step(self, self.action_map[action])

        new_player_coords = self._get_player_position()
        obs = self.render()
        
        # Show grid AFTER action
        logger.info(f"\nGrid AFTER action:")
        # logger.info(f"\n{obs}")
        logger.info(f"Player position after: {new_player_coords} (index: {int(player_pos)})")
        logger.info(f"Movement effective: {prev_player_position != int(player_pos)}")
        logger.info(f"Reward: {reward}, Done: {done}")
        logger.info("*" * 60)

        
        
        return obs, reward, done, {"action_is_effective": prev_player_position != int(player_pos)}

    def render(self, mode="encoded_image"):
        assert mode in ["encoded_image", "tiny_rgb_array", "list", "state", "rgb_array", "ansi", "dual"]
        
        # Handle default mode - save image and return encoded version
        if mode == "encoded_image":
            # Create images directory if it doesn't exist
            os.makedirs("images", exist_ok=True)
            
            # Get the rgb_array from gymnasium
            prev_render_mode = self.render_mode
            self.render_mode = "rgb_array"
            image_obs = GymFrozenLakeEnv.render(self)
            self.render_mode = prev_render_mode
            
            # Save the image to images folder
            image = Image.fromarray(image_obs)
            image_path = f"images/frozenlake_step_{self.step_count}.png"
            image.save(image_path)
            
            # Encode and return
            encoded_image = encode_image(image_obs)
            return encoded_image
        
        # Handle dual mode - returns both text and image
        if mode == "dual":
            text_obs = self.render(mode="tiny_rgb_array")
            encoded_image_obs = self.render(mode="encoded_image")
            
            return {
                "text": text_obs,
                "image": encoded_image_obs,
                "state": self.render(mode="state"),
                "list": self.render(mode="list")
            }
        
        if mode in ["rgb_array", "ansi"]:
            prev_render_mode = self.render_mode
            self.render_mode = mode
            obs = GymFrozenLakeEnv.render(self)
            self.render_mode = prev_render_mode
            return obs
        room_state = copy.deepcopy(self.desc)

        # replace the position of start 'S' with 'F'
        position_S = np.where(room_state == b"S")
        room_state[position_S] = b"F"

        # replace the position of the player with 'P'
        position_P = self._get_player_position()
        room_state[position_P] = b"P"

        if mode == "state":
            # transform 'S', 'F', 'H', 'G' to numpy integer array
            room_state = np.vectorize(lambda x: self.MAP_LOOKUP[x])(room_state)
            # add player in hole or player on goal
            if self.desc[position_P] == b"H":
                room_state[position_P] = 4
            elif self.desc[position_P] == b"G":
                room_state[position_P] = 5
            return room_state

        room_state = self.render(mode="state").tolist()

        if mode == "list":
            lookup = lambda cell: self.GRID_LOOKUP.get(cell, "?").strip("\t").strip()
            return [" ".join(lookup(cell) for cell in row) for row in room_state]

        if mode == "tiny_rgb_array":
            lookup = lambda cell: self.GRID_LOOKUP.get(cell, "?")
            result = "\n".join("".join(lookup(cell) for cell in row) for row in room_state)
            # result += f"Player Position is at ({position_P[0]}, {position_P[1]}), Goal Position is at ({self.goal_postion[0]}, {self.goal_postion[1]})"
            return result

    @staticmethod
    def from_dict(env_info: dict) -> "FrozenLakeEnv":
        return FrozenLakeEnv(size=env_info["size"], seed=env_info["seed"], p=env_info["p"], max_steps=env_info.get("max_steps", MAX_STEPS), is_slippery=env_info.get("is_slippery", False))
