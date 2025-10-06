import asyncio

from transformers import AutoTokenizer

from rllm.agents.frozenlake_agent import FrozenLakeAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.frozenlake.frozenlake import FrozenLakeEnv
from rllm.utils import compute_pass_at_k


def load_frozenlake_data():
    if DatasetRegistry.dataset_exists("frozenlake", "test"):
        test_dataset = DatasetRegistry.load_dataset("frozenlake", "test")
        return test_dataset.get_data()

    print("FrozenLake datasets not found. Preparing datasets...")
    from prepare_frozenlake_data import prepare_frozenlake_data

    train_dataset, test_dataset = prepare_frozenlake_data()

    return test_dataset.get_data()


if __name__ == "__main__":
    import os
    import logging
    import sys

    # Enable detailed logging to see what's happening inside
    logging.basicConfig(
        level=logging.INFO,  # Change to DEBUG for even more details
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout,  # Ensure output goes to stdout
        force=True  # Override any existing logging config
    )
    
    # Also set logger levels explicitly for our modules
    logging.getLogger('rllm.agents.frozenlake_agent').setLevel(logging.INFO)
    logging.getLogger('rllm.environments.frozenlake.frozenlake').setLevel(logging.INFO)
    logging.getLogger('rllm.engine.agent_execution_engine').setLevel(logging.INFO)
    
    print("=" * 80, flush=True)
    print("🚀 STARTING FROZENLAKE AGENT", flush=True)
    print("=" * 80, flush=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    n_parallel_agents = 1
    model_name = "unsloth/Qwen2-VL-7B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": model_name}

    max_steps = 20

    agent_args = {
        "max_steps": max_steps,
        "use_accumulate_history": True,
    }

    env_args = {
        "max_steps": max_steps,
        "is_slippery": False,
    }

    engine = AgentExecutionEngine(
        agent_class=FrozenLakeAgent,
        env_class=FrozenLakeEnv,
        agent_args=agent_args,
        env_args=env_args,
        engine_name="openai",
        tokenizer=tokenizer,
        max_steps=max_steps,
        sampling_params=sampling_params,
        rollout_engine_args={
            "base_url": "http://localhost:30000/v1",
            "api_key": "None",
        },
        max_response_length=2048,
        max_prompt_length=2048,
        n_parallel_agents=n_parallel_agents,
    )

    frozenlake_tasks = load_frozenlake_data()
    debugging_tasks = [task for task in frozenlake_tasks if task["size"] == 4]
    tasks = [debugging_tasks[0]]

    print(f"Num tasks: {len(tasks)}")

    # Process in small batches to avoid vLLM Metal KV cache corruption on Mac M4
    batch_size = 1
    all_results = []

    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        print(f"\n[INFO] Processing batch {i//batch_size + 1}/{(len(tasks) + batch_size - 1)//batch_size}")
        results = asyncio.run(engine.execute_tasks(batch))
        all_results.extend(results)


    print("\n" + "="*80)
    compute_pass_at_k(all_results)
