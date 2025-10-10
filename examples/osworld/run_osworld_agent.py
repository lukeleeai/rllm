"""Script to run end-to-end evaluation on OSWorld benchmark.
"""

import argparse
import asyncio
import json
import logging
import os
import sys

from transformers import AutoTokenizer

from rllm.agents.osworld_agents.scalecua_agent import ScaleCUA
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.osworld.desktop_env.desktop_env import DesktopEnv
from rllm.utils import compute_pass_at_k



def load_osworld_tasks(args):
    """Load OSWorld evaluation tasks from JSON configuration files."""
    with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
        test_all_meta = json.load(f)

    if args.domain != "all":
        test_all_meta = {args.domain: test_all_meta[args.domain]}

    # Filter out already completed tasks if result_dir exists
    if os.path.exists(args.result_dir):
        test_all_meta = get_unfinished(
            args.action_space,
            args.model,
            args.observation_type,
            args.result_dir,
            test_all_meta,
        )

    # Load all tasks
    tasks = []
    for domain, example_ids in test_all_meta.items():
        for example_id in example_ids:
            config_file = os.path.join(
                args.test_config_base_dir, f"examples/{domain}/{example_id}.json"
            )
            with open(config_file, "r", encoding="utf-8") as f:
                example = json.load(f)
                
            # Add metadata needed for environment setup
            result_dir = os.path.join(
                args.result_dir,
                str(args.model),
                domain,
                example_id,
            )

            print("Images will be saved to:", result_dir)
            
            task = {
                **example,
                "domain": domain,
                "example_id": example_id,
                "provider_name": args.provider_name,
                "region": None,  # Optional AWS region
                "path_to_vm": args.path_to_vm,
                "snapshot_name": example.get("snapshot", "chrome"),
                "action_space": args.action_space,
                "cache_dir": result_dir,  # Use result_dir as cache_dir
                "screen_size": (args.screen_width, args.screen_height),
                "headless": args.headless,
                "require_a11y_tree": args.observation_type in ["a11y_tree", "screenshot_a11y_tree", "som"],
                "require_terminal": False,  # Terminal not needed by default
                "os_type": "Ubuntu",
                "result_dir": result_dir,
            }
            tasks.append(task)

    return tasks


def get_unfinished(action_space, use_model, observation_type, result_dir, total_file_json):
    """Filter out already completed tasks."""
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)
    if not os.path.exists(target_dir):
        return total_file_json

    finished = {}
    for domain in os.listdir(target_dir):
        finished[domain] = []
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                if example_id == "onboard":
                    continue
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" not in os.listdir(example_path):
                        # Remove incomplete results
                        for file in os.listdir(example_path):
                            os.remove(os.path.join(example_path, file))
                    else:
                        finished[domain].append(example_id)

    if not finished:
        return total_file_json

    for domain, examples in finished.items():
        if domain in total_file_json:
            total_file_json[domain] = [
                x for x in total_file_json[domain] if x not in examples
            ]

    return total_file_json


def get_result(action_space, use_model, observation_type, result_dir):
    """Calculate and display current success rate."""
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)
    if not os.path.exists(target_dir):
        print("New experiment, no result yet.")
        return None

    all_result = []
    for domain in os.listdir(target_dir):
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" in os.listdir(example_path):
                        try:
                            result = float(
                                open(os.path.join(example_path, "result.txt"), "r").read()
                            )
                            all_result.append(result)
                        except:
                            all_result.append(0.0)

    if not all_result:
        print("New experiment, no result yet.")
        return None
    else:
        print(f"Current Success Rate: {sum(all_result) / len(all_result) * 100:.2f}%")
        return all_result


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on OSWorld benchmark"
    )

    # Environment config
    parser.add_argument("--provider_name", type=str, default="vmware", 
                       help="VM provider: vmware, virtualbox, docker, aws, etc.")
    parser.add_argument("--path_to_vm", type=str, default=None)
    parser.add_argument("--headless", action="store_true", 
                       help="Run in headless mode")
    parser.add_argument("--action_space", type=str, default="pyautogui", 
                       help="Action type")
    parser.add_argument("--observation_type",
                       choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
                       default="a11y_tree",
                       help="Observation type")
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=15)

    # Agent config
    parser.add_argument("--max_trajectory_length", type=int, default=3)
    parser.add_argument("--test_config_base_dir", type=str, 
                       default="evaluation_examples")

    # Model config
    parser.add_argument("--model", type=str, default="uitars")
    parser.add_argument("--model_type", type=str, default="qwen25vl")
    parser.add_argument("--api_url", type=str, default="http://localhost:30000/v1",
                       help="API URL for the model server")
    parser.add_argument("--max_pixels", type=float, default=2109744)
    parser.add_argument("--min_pixels", type=float, default=3136)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--history_n", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=500)

    # Task config
    parser.add_argument("--domain", type=str, default="all",
                       help="Domain to evaluate (or 'all' for all domains)")
    parser.add_argument("--test_all_meta_path", type=str, 
                       default="evaluation_examples/test_all.json")
    parser.add_argument("--result_dir", type=str, default="./results")
    parser.add_argument("--disable_think", action="store_true", 
                       help="Disable the think step")
    
    # Parallelization
    parser.add_argument("--n_parallel_agents", type=int, default=1,
                       help="Number of parallel agents/environments")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for processing tasks")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Enable detailed logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout,
        force=True
    )
    
    # Set logger levels for relevant modules
    logging.getLogger('rllm.agents.osworld_agents.scalecua_agent').setLevel(logging.INFO)
    logging.getLogger('rllm.environments.osworld.desktop_env').setLevel(logging.INFO)
    logging.getLogger('rllm.engine.agent_execution_engine').setLevel(logging.INFO)
    
    print("=" * 80, flush=True)
    print("🚀 STARTING OSWORLD AGENT EVALUATION", flush=True)
    print("=" * 80, flush=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Parse arguments
    args = config()
    
    print(f"\n📋 Configuration:")
    print(f"  Model: {args.model}")
    print(f"  API URL: {args.api_url}")
    print(f"  Domain: {args.domain}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Parallel agents: {args.n_parallel_agents}")
    print(f"  Action space: {args.action_space}")
    print(f"  Observation type: {args.observation_type}")
    print(f"  Result dir: {args.result_dir}")

    # Show current results if available
    print("\n" + "=" * 80)
    get_result(
        args.action_space,
        args.model,
        args.observation_type,
        args.result_dir,
    )
    
    # Load tasks
    print("\n" + "=" * 80)
    print("📂 Loading tasks...")
    tasks = load_osworld_tasks(args)
    print(f"✓ Loaded {len(tasks)} tasks")
    
    if len(tasks) == 0:
        print("✓ All tasks completed!")
        sys.exit(0)

    # Set up agent and environment arguments
    agent_args = {
        "model_name": args.model,
        "action_space": args.action_space,
        "observation_type": args.observation_type,
        "max_trajectory_length": args.max_trajectory_length,
        "model_type": args.model_type,
        "runtime_conf": {
            "history_n": args.history_n,
            "max_pixels": args.max_pixels,
            "min_pixels": args.min_pixels,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
        },
        "api_url": args.api_url,
        "max_steps": args.max_steps,
        "disable_think": args.disable_think,
        "use_accumulate_history": True,
    }

    env_args = {
        "provider_name": args.provider_name,
        "path_to_vm": args.path_to_vm,
        "action_space": args.action_space,
        "screen_size": (args.screen_width, args.screen_height),
        "headless": args.headless,
        "os_type": "Ubuntu",
        "setup_wait_time": 60,  # OSWorld needs 60 seconds after reset to be ready
    }

    # Set up tokenizer and processor for OSWorld agent
    # Tokenizer handles both text and vision tokens (e.g., <vision_start>, <image_pad>)
    # Processor handles actual image data (pixel values, preprocessing) - currently None for evaluation
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    processor = None  # Will be needed for VLM training in the future

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "model": args.model,
    }

    # Create execution engine
    print("\n" + "=" * 80)
    print("🔧 Initializing execution engine...")
    engine = AgentExecutionEngine(
        agent_class=ScaleCUA,
        env_class=DesktopEnv,
        agent_args=agent_args,
        env_args=env_args,
        engine_name="openai",
        tokenizer=tokenizer,
        processor=processor,
        max_steps=args.max_steps,
        sampling_params=sampling_params,
        rollout_engine_args={
            "base_url": args.api_url,
            "api_key": "None",
        },
        max_response_length=2048,
        max_prompt_length=2048,
        n_parallel_agents=args.n_parallel_agents,
    )
    print("✓ Engine initialized")

    # Process tasks in batches
    print("\n" + "=" * 80)
    print(f"🚀 Processing {len(tasks)} tasks in batches of {args.batch_size}...")
    
    all_results = []
    batch_size = args.batch_size

    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        batch_num = i//batch_size + 1
        total_batches = (len(tasks) + batch_size - 1)//batch_size
        
        print(f"\n[INFO] Processing batch {batch_num}/{total_batches} ({len(batch)} tasks)")
        
        try:
            results = asyncio.run(engine.execute_tasks(batch))
            all_results.extend(results)
            
            # Results are automatically saved by the engine to result_dir
            print(f"  ✓ Completed batch {batch_num}/{total_batches}")
                
        except Exception as e:
            print(f"  ✗ Error in batch {batch_num}: {e}")
            import traceback
            traceback.print_exc()

    # Compute final statistics
    print("\n" + "=" * 80)
    print("📊 EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Total tasks processed: {len(all_results)}")
    
    # Show updated results
    get_result(
        args.action_space,
        args.model,
        args.observation_type,
        args.result_dir,
    )
    
    # Compute pass@k if applicable
    if len(all_results) > 0:
        compute_pass_at_k(all_results)
