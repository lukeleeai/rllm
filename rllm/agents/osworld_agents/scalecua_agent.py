import ast
import base64
import logging
import math
import re
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Dict, List, Any
import copy

from PIL import Image
from openai import OpenAI

from osworld_agents.accessibility_tree_wrap.heuristic_retrieve import (
    filter_nodes,
)
from osworld_agents.prompts_single_action import (
    CUA_USER_PROMPT,
    CUA_SYSTEM_PROMPT_THOUGHT,
    CUA_SYSTEM_PROMPT_WITHOUT_THOUGHT,
)

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory

logger = logging.getLogger("desktopenv.agent")

FINISH_WORD = "finished"
WAIT_WORD = "wait"
ENV_FAIL_WORD = "error_env"
CALL_USER = "call_user"

IMAGE_FACTOR = 28
MIN_PIXELS = 3136
MAX_PIXELS = 2109744
MAX_RATIO = 200

pure_text_settings = ["a11y_tree"]

attributes_ns_ubuntu = "https://accessibility.windows.example.org/ns/attributes"
attributes_ns_windows = "https://accessibility.windows.example.org/ns/attributes"
state_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/state"
state_ns_windows = "https://accessibility.windows.example.org/ns/state"
component_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/component"
component_ns_windows = "https://accessibility.windows.example.org/ns/component"
value_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/value"
value_ns_windows = "https://accessibility.windows.example.org/ns/value"
class_ns_windows = "https://accessibility.windows.example.org/ns/class"
# More namespaces defined in OSWorld, please check desktop_env/server/main.py


def escape_single_quotes(text):
    pattern = r"(?<!\\)'"
    return re.sub(pattern, r"\\'", text)


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def encode_image(image_content):
    return base64.b64encode(image_content).decode("utf-8")


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def linear_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    if width * height > max_pixels:
        resize_factor = math.sqrt(max_pixels / (width * height))
        width, height = int(width * resize_factor), int(height * resize_factor)
    if width * height < min_pixels:
        resize_factor = math.sqrt(min_pixels / (width * height))
        width, height = math.ceil(width * resize_factor), math.ceil(
            height * resize_factor
        )

    return height, width


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def pil_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")  # 你可以改成 "JPEG" 等格式
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def linearize_accessibility_tree(accessibility_tree, platform="ubuntu"):

    if platform == "ubuntu":
        _attributes_ns = attributes_ns_ubuntu
        _state_ns = state_ns_ubuntu
        _component_ns = component_ns_ubuntu
        _value_ns = value_ns_ubuntu
    elif platform == "windows":
        _attributes_ns = attributes_ns_windows
        _state_ns = state_ns_windows
        _component_ns = component_ns_windows
        _value_ns = value_ns_windows
    else:
        raise ValueError("Invalid platform, must be 'ubuntu' or 'windows'")

    filtered_nodes = filter_nodes(ET.fromstring(accessibility_tree), platform)
    linearized_accessibility_tree = [
        "tag\tname\ttext\tclass\tdescription\tposition (top-left x&y)\tsize (w&h)"
    ]

    # Linearize the accessibility tree nodes into a table format
    for node in filtered_nodes:
        if node.text:
            text = (
                node.text
                if '"' not in node.text
                else '"{:}"'.format(node.text.replace('"', '""'))
            )

        elif node.get("{{{:}}}class".format(class_ns_windows), "").endswith(
            "EditWrapper"
        ) and node.get("{{{:}}}value".format(_value_ns)):
            node_text = node.get("{{{:}}}value".format(_value_ns), "")
            text = (
                node_text
                if '"' not in node_text
                else '"{:}"'.format(node_text.replace('"', '""'))
            )
        else:
            text = '""'

        linearized_accessibility_tree.append(
            "{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}".format(
                node.tag,
                node.get("name", ""),
                text,
                (
                    node.get("{{{:}}}class".format(_attributes_ns), "")
                    if platform == "ubuntu"
                    else node.get("{{{:}}}class".format(class_ns_windows), "")
                ),
                node.get("{{{:}}}description".format(_attributes_ns), ""),
                node.get("{{{:}}}screencoord".format(_component_ns), ""),
                node.get("{{{:}}}size".format(_component_ns), ""),
            )
        )

    return "\n".join(linearized_accessibility_tree)


def trim_accessibility_tree(linearized_accessibility_tree, max_tokens):
    # enc = tiktoken.encoding_for_model("gpt-4")
    # tokens = enc.encode(linearized_accessibility_tree)
    # if len(tokens) > max_tokens:
    #     linearized_accessibility_tree = enc.decode(tokens[:max_tokens])
    #     linearized_accessibility_tree += "[...]\n"
    return linearized_accessibility_tree


class ScaleCUA(BaseAgent):
    def __init__(
        self,
        model_name="uitars",
        platform="ubuntu",
        action_space="pyautogui",
        observation_type="screenshot",
        # observation_type can be in ["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"]
        max_trajectory_length=100,
        a11y_tree_max_tokens=10000,
        model_type="qwen25vl",
        runtime_conf: dict = {
            "history_n": 5,
            "max_pixels": 2109744,
            "min_pixels": 3136,
            "temperature": 1.0,
            "top_k": -1,
            "top_p": 0.9,
            "max_tokens": 500,
        },
        max_steps=15,
        api_url="http://127.0.0.1:8000/v1",
        disable_think=False,
        use_accumulate_history=True,
    ):
        # rLLM attribute
        self.messages = []
        self._trajectory = Trajectory()

        # ScaleCUA attributes
        self.model_name = model_name
        self.model = model_name
        self.max_steps = max_steps
        self.api_url = api_url
        self.platform = platform
        self.action_space = action_space
        self.observation_type = observation_type
        self.max_trajectory_length = max_trajectory_length
        self.a11y_tree_max_tokens = a11y_tree_max_tokens
        self.model_type = model_type
        self.runtime_conf = runtime_conf
        self.vlm = OpenAI(
            base_url=self.api_url,
            api_key="empty",
        )
        self.temperature = self.runtime_conf["temperature"]
        self.top_k = self.runtime_conf["top_k"]
        self.top_p = self.runtime_conf["top_p"]
        self.max_tokens = self.runtime_conf["max_tokens"]
        self.max_pixels = self.runtime_conf["max_pixels"]
        self.min_pixels = self.runtime_conf["min_pixels"]
        self.accumulate_history = use_accumulate_history

        self.thoughts = []
        self.actions = []
        self.observations = []
        self.history_images = []
        self.history_responses = []
        self.history_operation = []

        self.system_prompt_template = (
            CUA_SYSTEM_PROMPT_WITHOUT_THOUGHT
            if disable_think
            else CUA_SYSTEM_PROMPT_THOUGHT
        )
        self.user_prompt_template = CUA_USER_PROMPT

        if "history_n" in self.runtime_conf:
            self.history_n = self.runtime_conf["history_n"]
        else:
            self.history_n = 5

        self.cur_callusr_count = 0

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """
        Updates the agent's internal state after an environment step.
        Includes logic to check if the observation changed from the previous step.
        """
        logger.info("=" * 60)
        logger.info(f"STEP {self.step} - ENVIRONMENT UPDATE")
        logger.info("=" * 60)
        logger.info(f"Reward: {reward}, Done: {done}, Info: {info}")
        logger.info("=" * 60)

        user_prompt = ''

        # Check if the observation is the same as the previous step's observation
        # This check only makes sense if we have completed at least one step (i.e., received a model response and acted)
        if self._trajectory.steps and self._trajectory.steps[-1].action is not None:  # Check if the last step has an action (meaning it's a completed step)
            last_step_obs = self._trajectory.steps[-1].observation
            if last_step_obs == observation:
                user_prompt += "\nYour last response is invalid. Your position didn't change at all. You may need to recheck your thinking process, action outputted, and the format of response."

        screenshot = observation["screenshot"]
        screen_image = Image.open(BytesIO(screenshot))
        self.screen_width, self.screen_height = screen_image.size
    
        # Encode the screenshot bytes to base64
        screenshot_base64 = encode_image(screenshot)

        # if self.observation_type in ["screenshot", "screenshot_a11y_tree"]:
        #     base64_image = observation["screenshot"]
        #     try:
        #         linearized_accessibility_tree = (
        #             linearize_accessibility_tree(
        #                 accessibility_tree=observation["accessibility_tree"],
        #                 platform=self.platform,
        #             )
        #             if self.observation_type == "screenshot_a11y_tree"
        #             else None
        #         )
        #     except:
        #         linearized_accessibility_tree = None

        #     if linearized_accessibility_tree:
        #         linearized_accessibility_tree = trim_accessibility_tree(
        #             linearized_accessibility_tree, self.a11y_tree_max_tokens
        #         )

        #     if self.observation_type == "screenshot_a11y_tree":
        #         self.observations.append(
        #             {
        #                 "screenshot": base64_image,
        #                 "accessibility_tree": linearized_accessibility_tree,
        #             }
        #         )
        #     else:
        #         self.observations.append(
        #             {"screenshot": base64_image, "accessibility_tree": None}
        #         )

        # else:
        #     raise ValueError(
        #         "Invalid observation_type type: " + self.observation_type
        #     )

        # byte_image = self.history_images[-1]
        # cur_image = Image.open(BytesIO(byte_image))
        # self.screen_width, self.screen_height = cur_image.size
        # action_history = self.format_history(self.history_operation)
        # # TODO: add a task instruction here
        # user_prompt = self.user_prompt_template.format(
        #     instruction="",
        #     history=action_history,
        # )

        # Add the user message for the *next* interaction turn
        self.messages.append({"role": "user", "content": [ # this results in an error when converting to tokens
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{screenshot_base64}"
                },
            },
            {"type": "text", "text": user_prompt},
        ]})

        self.current_observation = observation

    def update_from_model(self, response, **kwargs) -> Action:
        """
        Updates the agent's internal state after the model generates a response.
        """
        print(f"Response: {response}")
        
        # Handle error strings returned by the engine
        if isinstance(response, str) and response.startswith("Error"):
            logger.error(f"Received error response: {response}")
            return Action(action="DONE")
        
        # Response is the content string from the model
        prediction = response.strip() if isinstance(response, str) else response
        thought, operation, actions = self.parse_response(prediction)

        if prediction is None:
            return Action(action="DONE")

        logger.info(f"🤔 Agent's Thought Process:\n{thought}")
        logger.info(f"🎯 Parsed Action: {actions}")
        logger.info("-" * 60)

        try:
            pyautogui_actions = self.parse_action(actions)
        except Exception as e:
            print(f"Parsing action error: {prediction}, with error:\n{e}")
            return Action(action="DONE")

        # Check if agent indicates task completion
        # TODO: how to handle multiple actions?
        completion_keywords = ["completed", "finished", "done", "successfully", "success"]
        if (not pyautogui_actions or pyautogui_actions == []) and any(
            keyword in prediction.lower() for keyword in completion_keywords
        ):
            logger.info("Agent indicated task completion - returning DONE")
            pyautogui_actions = ["DONE"]

        if self.step >= self.max_steps:
            # Default to FAIL if exceed max steps
            pyautogui_actions = ["FAIL"]
        # self.actions.append(pyautogui_actions)

        new_step = Step(chat_completions=copy.deepcopy(self.chat_completions), thought=thought, action=actions, model_response=prediction, observation=self.current_observation)
        self._trajectory.steps.append(new_step)

        print("Actions: ", actions)
        self.messages.append({"role": "assistant", "content": f"{thought}\n\n{actions}"})

        self.step += 1

        logger.info("RESPONE: %s", repr(prediction))
        logger.info("PARSED ACTION: %s", repr(actions))
        logger.info("pyautogui code(s): %s", repr(pyautogui_actions))
        
        # Return the first pyautogui action (executable code string) for the environment
        # The environment expects a single action string or ["DONE"]/["FAIL"] etc.
        return Action(action=pyautogui_actions[0] if pyautogui_actions else "DONE")
    
    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """
        Returns the chat completions of the agent.
        """
        if self.accumulate_history:
            return self.messages
        else:
            if len(self.messages) <= 1:
                return self.messages
            else:
                return [self.messages[0], self.messages[-1]]

    @property
    def trajectory(self) -> Trajectory:
        """
        Returns the trajectory of the agent.
        """
        return self._trajectory
    
    def reset(self, _logger=None):
        global logger
        logger = (
            _logger if _logger is not None else logging.getLogger("desktopenv.agent")
        )

        self.thoughts = []
        self.actions = []
        self.observations = []

        self.history_images = []
        self.history_responses = []
        self.history_operation = [
            {
                "role": "system",
                "content": self.system_prompt_template,
            }
        ]

        self._trajectory = Trajectory()
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt_template,
            }
        ]
        self.step = 0


    def format_history(self, history):
        if len(history) > 0:
            actions_history = [
                f"Step {i+1}: {low_level}" for i, low_level in enumerate(history)
            ]
        else:
            actions_history = None
        return "\n".join(actions_history) if actions_history is not None else None

    def predict(
        self, instruction: str, obs: Dict, last_action_after_obs: Dict = None
    ) -> List:
        """
        Predict the next action(s) based on the current observation.
        """
        assert len(self.observations) == len(self.actions) and len(self.actions) == len(
            self.thoughts
        ), "The number of observations and actions should be the same."

        if len(self.observations) > self.max_trajectory_length:
            if self.max_trajectory_length == 0:
                _observations = []
                _actions = []
                _thoughts = []
            else:
                _observations = self.observations[-self.max_trajectory_length :]
                _actions = self.actions[-self.max_trajectory_length :]
                _thoughts = self.thoughts[-self.max_trajectory_length :]
        else:
            _observations = self.observations
            _actions = self.actions
            _thoughts = self.thoughts

        for previous_obs, previous_action, previous_thought in zip(
            _observations, _actions, _thoughts
        ):
            # {{{1
            if self.observation_type in ["screenshot_a11y_tree", "screenshot"]:
                _screenshot = previous_obs["screenshot"]
                _linearized_accessibility_tree = previous_obs["accessibility_tree"]

            else:
                raise ValueError(
                    "Invalid observation_type type: " + self.observation_type
                )  # 1}}}

        self.history_images.append(obs["screenshot"])

        if self.observation_type in ["screenshot", "screenshot_a11y_tree"]:
            base64_image = obs["screenshot"]
            try:
                linearized_accessibility_tree = (
                    linearize_accessibility_tree(
                        accessibility_tree=obs["accessibility_tree"],
                        platform=self.platform,
                    )
                    if self.observation_type == "screenshot_a11y_tree"
                    else None
                )
            except:
                linearized_accessibility_tree = None
            # logger.debug("LINEAR AT: %s", linearized_accessibility_tree)

            if linearized_accessibility_tree:
                linearized_accessibility_tree = trim_accessibility_tree(
                    linearized_accessibility_tree, self.a11y_tree_max_tokens
                )

            if self.observation_type == "screenshot_a11y_tree":
                self.observations.append(
                    {
                        "screenshot": base64_image,
                        "accessibility_tree": linearized_accessibility_tree,
                    }
                )
            else:
                self.observations.append(
                    {"screenshot": base64_image, "accessibility_tree": None}
                )

        else:
            raise ValueError(
                "Invalid observation_type type: " + self.observation_type
            )  # 1}}}

        byte_image = self.history_images[-1]
        cur_image = Image.open(BytesIO(byte_image))
        self.screen_width, self.screen_height = cur_image.size
        action_history = self.format_history(self.history_operation)
        user_prompt = self.user_prompt_template.format(
            instruction=instruction,
            history=action_history,
        )
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt_template}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encode_image(byte_image)}"
                        },
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        #########################################################################################

        # update_from_model

        #########################################################################################

        try_times = 3
        while True:
            if try_times <= 0:
                print(
                    f"Reach max retry times to fetch response from client, as error flag."
                )
                return "client error", ["DONE"]
            try:
                response = self.vlm.chat.completions.create(
                    model=self.vlm.models.list().data[0].id,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=1,
                    top_p=self.top_p,
                )
                # print(response.choices[0].message.content)
                prediction = response.choices[0].message.content.strip()
                think, operation, actions = self.parse_response(prediction)
                break
                # prediction = response[0]["prediction"].strip()
            except Exception as e:
                print(
                    f"Error when fetching response from client, with response: {response}"
                )
                prediction = None
                try_times -= 1

        if prediction is None:
            return "client error", ["DONE"]

        self.history_responses.append(prediction)
        self.history_operation.append(operation)
        self.thoughts.append(think)

        try:
            pyautogui_actions = self.parse_action(actions)
        except Exception as e:
            print(f"Parsing action error: {prediction}, with error:\n{e}")
            return f"Parsing action error: {prediction}, with error:\n{e}", ["DONE"]

        # Check if agent indicates task completion
        completion_keywords = ["completed", "finished", "done", "successfully", "success"]
        if (not pyautogui_actions or pyautogui_actions == []) and any(
            keyword in prediction.lower() for keyword in completion_keywords
        ):
            logger.info("Agent indicated task completion - returning DONE")
            pyautogui_actions = ["DONE"]

        if len(self.history_responses) >= self.max_steps:
            # Default to FAIL if exceed max steps
            pyautogui_actions = ["FAIL"]
        self.actions.append(pyautogui_actions)

        logger.info("RESPONE: %s", repr(prediction))
        logger.info("PARSED ACTION: %s", repr(actions))
        logger.info("pyautogui code(s): %s", repr(pyautogui_actions))
        return prediction, pyautogui_actions

    def parse_response(self, response: str) -> Dict:
        action_matches = re.findall(
            r"<action>\s*(.*?)\s*</action>", response, re.DOTALL
        )
        actions = []
        if action_matches:
            for match in action_matches:
                # Split each match by newline and strip whitespace from each line
                lines = [line.strip() for line in match.split("\n") if line.strip()]
                actions.extend(lines)
        # actions = [action.strip() for action in action_matches] if action_matches else None
        # click
        # type
        operation_match = re.search(
            r"<operation>\s*(.*?)\s*</operation>", response, re.DOTALL
        )
        operation = operation_match.group(1).strip() if operation_match else None

        think_match = re.search(r"<think>\s*(.*?)\s*</think>", response, re.DOTALL)
        think = think_match.group(1).strip() if think_match else None

        return (think, operation, actions)

    def parse_action(self, actions):
        parsed_action = []
        for action in actions:
            match = re.match(r"(\w+)\((.*)\)", action)
            if not match:
                return None

            func_name = match.group(1)
            args_str = match.group(2)
            args = {}

            if "hotkey" in func_name.lower():
                keys = re.findall(r"'(.*?)'", args_str)
                keys = [key.lower() for key in keys]
                args["args"] = keys
            elif "press" in func_name.lower():
                keys = None
                presses = 1
                presses_match = re.search(r"presses\s*=\s*(\d+)", args_str)
                if presses_match:
                    presses = int(presses_match.group(1))
                    args_str = (
                        args_str[: presses_match.start()]
                        + args_str[presses_match.end() :]
                    )
                    args_str = args_str.rstrip(", ").strip()

                keys_keyword_match = re.search(r"keys\s*=\s*(.*)", args_str, re.DOTALL)
                if keys_keyword_match:
                    keys_str = keys_keyword_match.group(1).strip()
                    if (keys_str.startswith("'") and keys_str.endswith("'")) or (
                        keys_str.startswith('"') and keys_str.endswith('"')
                    ):
                        keys_str = keys_str[1:-1]
                    elif keys_str.startswith("[") and keys_str.endswith("]"):

                        keys_str = ast.literal_eval(keys_str)
                    keys = keys_str
                elif args_str:
                    keys_str = args_str.strip()
                    if (keys_str.startswith("'") and keys_str.endswith("'")) or (
                        keys_str.startswith('"') and keys_str.endswith('"')
                    ):
                        keys_str = keys_str[1:-1]
                    keys = keys_str

                args["keys"] = keys
                args["presses"] = presses
            elif "scroll" in func_name.lower():
                clicks, x, y = None, None, None
                if "=" in args_str:
                    kwargs = dict(re.findall(r"(\w+)\s*=\s*(-?\d+)", args_str))

                    clicks = (
                        int(kwargs.get("clicks"))
                        if kwargs.get("clicks") is not None
                        else None
                    )
                    x = int(kwargs.get("x")) if kwargs.get("x") is not None else None
                    y = int(kwargs.get("y")) if kwargs.get("y") is not None else None

                elif args_str:
                    try:
                        clicks = int(args_str)
                    except ValueError:
                        pass

                if clicks:
                    args["clicks"] = clicks
                if x:
                    args["x"] = x
                if y:
                    args["y"] = y

            else:
                if "=" in args_str:
                    for arg in re.finditer(r"(\w+)=\[([^\]]+)\]", args_str):
                        param = arg.group(1)
                        list_str = arg.group(2)

                        list_items = []
                        for item in re.finditer(
                            r"'([^']*)'|\"([^\"]*)\"|([^,\]]+)", list_str
                        ):
                            val = (
                                item.group(1) or item.group(2) or item.group(3)
                            ).strip()
                            if val:
                                list_items.append(val.strip("\"'"))

                        args[param] = list_items

                    for arg in re.finditer(r"(\w+)=([^,)]+)", args_str):
                        param = arg.group(1)
                        if param in args:
                            continue

                        value_str = arg.group(2).strip()

                        if value_str.isdigit():
                            value = int(value_str)
                        elif value_str.replace(".", "", 1).isdigit():
                            value = float(value_str)
                        elif value_str.lower() in ("true", "false"):
                            value = value_str.lower() == "true"
                        else:
                            value = value_str.strip("\"'")

                        args[param] = value

                else:
                    args_list = []
                    for arg in re.finditer(r"'([^']*)'|\"([^\"]*)\"|([^,]+)", args_str):
                        val = (arg.group(1) or arg.group(2) or arg.group(3)).strip()
                        if val:
                            args_list.append(val.strip("\"'"))

                    if args_list:
                        args["args"] = args_list

            parsed_action.append({"name": func_name, "parameters": args})

        pyautogui_actions = []
        for action in parsed_action:
            logger.info(
                f"************************ Parsed Action ************************"
            )
            logger.info(action)
            pyautogui_actions.append(self.transform_action(action))

        return pyautogui_actions

    def _parse_kwargs(self, arg_str: str) -> Dict[str, str]:
        try:
            parsed = ast.literal_eval(arg_str)
            return {k: str(v) for k, v in parsed.items()}
        except Exception as e:
            print(f"Failed to parse argument string: {arg_str}, error: {e}")
            return {}

    def transform_action(self, action) -> str:
        """Convert raw action signature to executable code using pyautogui."""
        func, arg_str = action["name"], str(action["parameters"])
        kwargs = self._parse_kwargs(arg_str)

        if func in ["click", "doubleClick", "rightClick"]:
            x = kwargs.get("x")
            y = kwargs.get("y")
            resize_h, resize_w = smart_resize(
                self.screen_height,
                self.screen_width,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            x = "{:.4f}".format(float(x) / resize_w * self.screen_width)
            y = "{:.4f}".format(float(y) / resize_h * self.screen_height)
            clicks = "2" if func == "doubleClick" else kwargs.get("clicks", "1")
            button = (
                '"right"' if func == "rightClick" else kwargs.get("button", '"left"')
            )
            return f"pyautogui.click(x={x}, y={y}, clicks={clicks}, button={button})"

        if func == "scroll":
            clicks = kwargs.get("clicks", "0")
            x = kwargs.get("x", None)
            y = kwargs.get("y", None)
            if x is not None and y is not None:
                resize_h, resize_w = smart_resize(
                    self.screen_height,
                    self.screen_width,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                )
                x = "{:.4f}".format(float(x) / resize_w * self.screen_width)
                y = "{:.4f}".format(float(y) / resize_h * self.screen_height)
                return f"pyautogui.scroll({clicks}, x={x}, y={y})"
            else:
                return f"pyautogui.scroll({clicks})"

        # moveTo
        if func == "moveTo":
            x = kwargs.get("x")
            y = kwargs.get("y")
            resize_h, resize_w = smart_resize(
                self.screen_height,
                self.screen_width,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            x = "{:.4f}".format(float(x) / resize_w * self.screen_width)
            y = "{:.4f}".format(float(y) / resize_h * self.screen_height)
            return f"pyautogui.moveTo({x}, {y})"

        # dragTo
        if func == "dragTo":
            x = kwargs.get("x")
            y = kwargs.get("y")
            button = kwargs.get("button", '"left"')
            resize_h, resize_w = smart_resize(
                self.screen_height,
                self.screen_width,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            x = "{:.4f}".format(float(x) / resize_w * self.screen_width)
            y = "{:.4f}".format(float(y) / resize_h * self.screen_height)
            return f"pyautogui.dragTo({x}, {y}, button={button})"

        # press
        if func == "press":
            key = kwargs.get("keys", None)
            if key is None:
                key = kwargs.get("key")
            try:
                key = ast.literal_eval(key)
            except (ValueError, SyntaxError):
                pass
            presses = int(kwargs.get("presses", 1))
            if isinstance(key, str):
                return "; ".join([f"pyautogui.press('{key}')" for _ in range(presses)])
            elif isinstance(key, list):
                lines = []
                for _ in range(presses):
                    for k in key:
                        lines += [f"pyautogui.press('{k}')"]
                return "; ".join(lines)

        # hotkey
        if func == "hotkey":

            args_dict = ast.literal_eval(arg_str)
            keys = args_dict["args"]
            args_str = ", ".join(f'"{k}"' for k in keys)
            return f"pyautogui.hotkey({args_str})"

        # keyDown
        if func == "keyDown":
            key = kwargs.get("key")
            return f'pyautogui.keyDown("{key}")'

        # keyUp
        if func == "keyUp":
            key = kwargs.get("key")
            return f'pyautogui.keyUp("{key}")'

        # write
        if func == "write":
            msg = kwargs.get("message") or kwargs.get("msg") or ""
            return f'pyautogui.write("{msg}")'

        # wait
        if func == "wait":
            secs = kwargs.get("seconds", "6")
            return f"time.sleep({secs})"

        # terminate
        if func == "terminate":
            return "DONE"

        return f"# Unhandled action: {func}"


if __name__ == "__main__":
    agent = ScaleCUA()
    print(agent.parse_action(["hotkey(['cmd','a'])"]))
    print(agent.parse_action(["hotkey('cmd','a')"]))