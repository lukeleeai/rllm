from __future__ import annotations

import logging
import os
import time
from typing import Callable, Any, Optional, Tuple
from typing import List, Dict, Union

import gymnasium as gym
from PIL import Image

from desktop_env.controllers.python import PythonController
from desktop_env.controllers.setup import SetupController
from desktop_env.evaluators import metrics, getters
from desktop_env.providers import create_vm_manager_and_provider

from rllm.environments.base.base_env import BaseEnv

logger = logging.getLogger("desktopenv.env")

Metric = Callable[[Any, Any], float]
Getter = Callable[[gym.Env, Dict[str, Any]], Any]


class DesktopEnv(BaseEnv):
    """
    DesktopEnv with OpenAI Gym interface. It provides a desktop environment for setting and evaluating desktop automation tasks.
    """

    def __init__(
        self,
        provider_name: str = "docker",
        region: str = None,
        path_to_vm: str = None,
        snapshot_name: str = "init_state",
        action_space: str = "computer_13",
        cache_dir: str = "cache",
        screen_size: Tuple[int] = (1920, 1080),
        headless: bool = False,
        require_a11y_tree: bool = True,
        require_terminal: bool = False,
        os_type: str = "Ubuntu",
        setup_wait_time: int = 0,
        result_dir: str = None,
    ):
        """
        Args:
            provider_name (str): virtualization provider name, default to "vmware"
            region (str): the region for allocate machines, work for cloud services, default to  "us-east-1"
            path_to_vm (str): path to .vmx file
            snapshot_name (str): snapshot name to revert to, default to "init_state"
            action_space (str): "computer_13" | "pyautogui"
            cache_dir (str): cache directory to cache task-related stuffs like
              reference file for evaluation
            screen_size (Tuple[int]): screen size of the VM
            headless (bool): whether to run the VM in headless mode
            require_a11y_tree (bool): whether to require accessibility tree
            require_terminal (bool): whether to require terminal output
            setup_wait_time (int): time to wait after reset for environment to be ready (seconds)
            result_dir (str): directory to save screenshots and recording
        """
        # Initialize VM manager and vitualization provider
        self.region = region

        # Default
        self.server_port = 5000
        self.chromium_port = 9222
        self.vnc_port = 5900
        self.vlc_port = 8080
        self.manager, self.provider = create_vm_manager_and_provider(
            provider_name, region
        )

        self.os_type = os_type

        # Initialize environment variables
        if path_to_vm:
            self.path_to_vm = (
                os.path.abspath(os.path.expandvars(os.path.expanduser(path_to_vm)))
                if provider_name in {"vmware", "virtualbox"}
                else path_to_vm
            )
        else:
            self.path_to_vm = self.manager.get_vm_path(self.os_type, region)

        self.snapshot_name = snapshot_name
        self.cache_dir_base: str = cache_dir
        # todo: add the logic to get the screen size from the VM
        self.headless = headless
        self.require_a11y_tree = require_a11y_tree
        self.require_terminal = require_terminal
        self.setup_wait_time = setup_wait_time
        self.result_dir = result_dir
        self.recording_started = False

        # Initialize emulator and controller
        if (
            provider_name != "docker"
        ):  # Check if this is applicable to other VM providers
            logger.info("Initializing...")
            self._start_emulator()

        # mode: human or machine
        self.instruction = None
        assert action_space in ["computer_13", "pyautogui"]
        self.ACTION_SPACE = action_space  # todo: refactor it to the ActType

        # episodic stuffs, like counters, will be updated or reset
        # when calling self.reset()
        self.step_count: int = 0
        self.action_history: List[Dict[str, any]] = []

    def _start_emulator(self):
        # Power on the virtual machine
        self.provider.start_emulator(self.path_to_vm, self.headless, self.os_type)

        # Get the ip from the virtual machine, and setup the controller
        vm_ip_ports = self.provider.get_ip_address(self.path_to_vm).split(":")
        self.vm_ip = vm_ip_ports[0]
        if len(vm_ip_ports) > 1:
            self.server_port = int(vm_ip_ports[1])
            self.chromium_port = int(vm_ip_ports[2])
            self.vnc_port = int(vm_ip_ports[3])
            self.vlc_port = int(vm_ip_ports[4])
        print(self.vlc_port)
        # import ipdb;ipdb.set_trace()
        self.controller = PythonController(
            vm_ip=self.vm_ip, server_port=self.server_port
        )
        self.setup_controller = SetupController(
            vm_ip=self.vm_ip,
            server_port=self.server_port,
            chromium_port=self.chromium_port,
            vlc_port=self.vlc_port,
            cache_dir=self.cache_dir_base,
        )

    def _revert_to_snapshot(self):
        # Revert to certain snapshot of the virtual machine, and refresh the path to vm and ip of vm
        # due to the fact it could be changed when implemented by cloud services
        path_to_vm = self.provider.revert_to_snapshot(
            self.path_to_vm, self.snapshot_name
        )
        if path_to_vm and not path_to_vm == self.path_to_vm:
            # path_to_vm has to be a new path
            self.manager.delete_vm(self.path_to_vm, self.region)
            self.manager.add_vm(path_to_vm, self.region)
            self.manager.occupy_vm(path_to_vm, os.getpid(), self.region)
            self.path_to_vm = path_to_vm

    def _save_state(self, snapshot_name=None):
        # Save the current virtual machine state to a certain snapshot name
        self.provider.save_state(self.path_to_vm, snapshot_name)

    def close(self):
        # Close (release) the virtual machine
        self.provider.stop_emulator(self.path_to_vm)

    def reset(
        self, task_config: Optional[Dict[str, Any]] = None, seed=None, options=None
    ) -> Dict[str, Any]:
        # Reset to certain task in OSWorld
        logger.info("Resetting environment...")
        logger.info("Switching task...")
        logger.info("Setting counters...")
        self.step_count = 0
        self.action_history.clear()

        logger.info("Reverting to snapshot to {}...".format(self.snapshot_name))
        self._revert_to_snapshot()
        logger.info("Starting emulator...")
        self._start_emulator()
        logger.info("Emulator started.")

        if task_config is not None:
            self._set_task_info(task_config)
            self.setup_controller.reset_cache_dir(self.cache_dir)
            logger.info("Setting up environment...")
            try:
                self.setup_controller.setup(self.config)
                logger.info("Environment setup complete.")
            except Exception as e:
                logger.error(f"Error during environment setup: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise

        time.sleep(2)
        
        # Wait for environment to be ready (OS-specific setup time)
        if self.setup_wait_time > 0:
            logger.info(f"Waiting {self.setup_wait_time} seconds for environment to be ready...")
            # time.sleep(self.setup_wait_time)  # TODO: Bring it back after testing
            logger.info("Wait complete.")
        
        # Start recording if controller supports it
        if hasattr(self, 'controller') and hasattr(self.controller, 'start_recording'):
            logger.info("Starting recording...")
            self.controller.start_recording()
            self.recording_started = True
            logger.info("Recording started.")
        
        observation = self._get_obs()
        return observation, {}

    def _get_obs(self):
        # We provide screenshot, accessibility_tree (optional), terminal (optional), and instruction.
        # can be customized and scaled
        screenshot = self.controller.get_screenshot()
        
        # Call render to save screenshot if result_dir is configured
        self.render(mode="rgb_array", screenshot=screenshot)
        
        return {
            "screenshot": screenshot,
            "accessibility_tree": (
                self.controller.get_accessibility_tree()
                if self.require_a11y_tree
                else None
            ),
            "terminal": (
                self.controller.get_terminal_output() if self.require_terminal else None
            ),
            "instruction": self.instruction,
        }

    @property
    def vm_platform(self):
        return self.controller.get_vm_platform()

    @property
    def vm_screen_size(self):
        return self.controller.get_vm_screen_size()

    def _set_task_info(self, task_config: Dict[str, Any]):
        self.task_id: str = task_config["id"]
        self.cache_dir: str = os.path.join(self.cache_dir_base, self.task_id)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.instruction = task_config["instruction"]
        self.config = task_config["config"] if "config" in task_config else []

        # evaluator dict
        # func -> metric function string, or list of metric function strings
        # conj -> conjunction of multiple metrics if func is a list with length > 1, "and"/"or"
        # result -> result getter config, or list of result getter configs
        # expected (optional) -> expected getter config, or list of expected getter configs
        # options (optional) -> metric options, or list of metric options
        # if func is a str list, then result, expected (if exists), options (if exists) should also be lists of the same length
        # even if one of the metrics does not need expected or options field, it should be included in the list with None
        self.evaluator = task_config["evaluator"]
        self.metric: Metric = (
            [getattr(metrics, func) for func in self.evaluator["func"]]
            if isinstance(self.evaluator["func"], list)
            else getattr(metrics, self.evaluator["func"])
        )
        self.metric_conj: str = self.evaluator.get(
            "conj", "and"
        )  # take conjunction of multiple metrics
        if "result" in self.evaluator and len(self.evaluator["result"]) > 0:
            self.result_getter: Getter = (
                [
                    getattr(getters, "get_{:}".format(res["type"]))
                    for res in self.evaluator["result"]
                ]
                if isinstance(self.evaluator["result"], list)
                else getattr(
                    getters, "get_{:}".format(self.evaluator["result"]["type"])
                )
            )
        else:
            self.result_getter = (
                [None] * len(self.metric) if isinstance(self.metric, list) else None
            )

        if "expected" in self.evaluator and len(self.evaluator["expected"]) > 0:
            self.expected_getter: Getter = (
                [
                    getattr(getters, "get_{:}".format(exp["type"])) if exp else None
                    for exp in self.evaluator["expected"]
                ]
                if isinstance(self.evaluator["expected"], list)
                else getattr(
                    getters, "get_{:}".format(self.evaluator["expected"]["type"])
                )
            )
        else:
            self.expected_getter = (
                [None] * len(self.metric) if isinstance(self.metric, list) else None
            )
        self.metric_options: Union[List[Dict[str, Any]], Dict[str, Any]] = (
            [opt if opt else {} for opt in self.evaluator["options"]]
            if isinstance(self.evaluator.get("options", {}), list)
            else (
                self.evaluator["options"]
                if "options" in self.evaluator
                else [{}] * len(self.metric) if isinstance(self.metric, list) else {}
            )
        )

        assert not isinstance(self.evaluator["func"], list) or (
            len(self.metric)
            == len(self.result_getter)
            == len(self.expected_getter)
            == len(self.metric_options)
        )

    def step(self, action, pause=2):
        self.step_count += 1
        self.action_history.append(action)

        reward = 0  # todo: Define reward calculation for each example
        done = False  # todo: Define episode termination condition for each example
        info = {}

        # handle the special actions
        if action in ["WAIT", "FAIL", "DONE"] or (
            type(action) == dict and action["action_type"] in ["WAIT", "FAIL", "DONE"]
        ):
            if action == "WAIT":
                time.sleep(pause)
            elif action == "FAIL":
                done = True
                info = {"fail": True}
            elif action == "DONE":
                done = True
                info = {"done": True}

        if self.ACTION_SPACE == "computer_13":
            # the set of all possible actions defined in the action representation
            self.controller.execute_action(action)
        elif self.ACTION_SPACE == "pyautogui":
            if action in ["WAIT", "FAIL", "DONE"]:
                self.controller.execute_action(action)
            else:
                # the set of all possible python commands insides `pyautogui`
                self.controller.execute_python_command(action)

        time.sleep(pause)
        observation = self._get_obs()
        
        # End recording if episode is done and recording was started
        if done and self.recording_started:
            if hasattr(self, 'controller') and hasattr(self.controller, 'end_recording'):
                recording_path = os.path.join(self.result_dir, "recording.mp4") if self.result_dir else "recording.mp4"
                logger.info(f"Ending recording and saving to {recording_path}...")
                self.controller.end_recording(recording_path)
                self.recording_started = False
                logger.info("Recording ended.")

        return observation, reward, done, info

    def evaluate(self):
        """
        Evaluate whether the task is successfully completed.
        """

        self.setup_controller.setup(self.evaluator.get("postconfig", []))

        if self.evaluator["func"] == "infeasible":
            if len(self.action_history) > 0 and self.action_history[-1] == "FAIL":
                return 1
            else:
                return 0
        else:
            if len(self.action_history) > 0 and self.action_history[-1] == "FAIL":
                return 0

        if type(self.metric) == list:
            # Multiple metrics to evaluate whether the task is successfully completed
            results = []
            assert len(self.metric) == len(
                self.result_getter
            ), "The number of metrics and result getters must be the same"
            if "expected" in self.evaluator:
                assert len(self.metric) == len(
                    self.expected_getter
                ), "The number of metrics and expected getters must be the same"
            for idx, metric in enumerate(self.metric):
                try:
                    config = self.evaluator["result"][idx]
                    result_state = self.result_getter[idx](self, config)
                except FileNotFoundError:
                    logger.error("File not found!")
                    if self.metric_conj == "and":
                        return 0

                if (
                    "expected" in self.evaluator
                    and self.expected_getter
                    and self.evaluator["expected"]
                ):
                    expected_state = self.expected_getter[idx](
                        self, self.evaluator["expected"][idx]
                    )
                    metric: int = metric(
                        result_state, expected_state, **self.metric_options[idx]
                    )
                else:
                    metric: int = metric(result_state, **self.metric_options[idx])

                if self.metric_conj == "and" and float(metric) == 0.0:
                    return 0
                elif self.metric_conj == "or" and float(metric) == 1.0:
                    return 1
                else:
                    results.append(metric)

            return (
                sum(results) / len(results)
                if self.metric_conj == "and"
                else max(results)
            )
        else:
            # Single metric to evaluate whether the task is successfully completed
            try:
                result_state = self.result_getter(self, self.evaluator["result"])
            except FileNotFoundError:
                logger.error("File not found!")
                return 0

            if (
                "expected" in self.evaluator
                and self.expected_getter
                and self.evaluator["expected"]
            ):
                expected_state = self.expected_getter(self, self.evaluator["expected"])
                metric: float = self.metric(
                    result_state, expected_state, **self.metric_options
                )
            else:
                metric: float = self.metric(result_state, **self.metric_options)

        return metric

    def render(self, mode="rgb_array", screenshot=None):
        """Save screenshot to result_dir if configured.
        
        Args:
            mode: Rendering mode ("rgb_array" supported)
            screenshot: Optional screenshot bytes to save (if None, gets from controller)
        """
        if mode == "rgb_array":
            if screenshot is None:
                screenshot = self.controller.get_screenshot()
            
            # Save screenshot if result_dir is configured
            if self.result_dir:
                import datetime
                os.makedirs(self.result_dir, exist_ok=True)
                
                # Generate timestamp for step screenshots (not for initial step_0)
                if self.step_count > 0:
                    action_timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
                    screenshot_path = os.path.join(
                        self.result_dir, 
                        f"step_{self.step_count}_{action_timestamp}.png"
                    )
                else:
                    # Initial screenshot without timestamp
                    screenshot_path = os.path.join(self.result_dir, f"step_{self.step_count}.png")
                
                with open(screenshot_path, "wb") as f:
                    f.write(screenshot)
            
            return screenshot
        else:
            raise ValueError("Unsupported render mode: {}".format(mode))

    @staticmethod
    def from_dict(env_info: dict) -> "DesktopEnv":
        return DesktopEnv(
            provider_name=env_info["provider_name"],
            region=env_info["region"],
            path_to_vm=env_info["path_to_vm"],
            snapshot_name=env_info["snapshot_name"],
            action_space=env_info["action_space"],
            cache_dir=env_info["cache_dir"],
            screen_size=env_info["screen_size"],
            headless=env_info["headless"],
            require_a11y_tree=env_info["require_a11y_tree"],
            require_terminal=env_info["require_terminal"],
            os_type=env_info["os_type"],
            setup_wait_time=env_info.get("setup_wait_time", 0),
            result_dir=env_info.get("result_dir", None),
        )