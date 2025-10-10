#!/bin/bash

# Run OSWorld agent evaluation with ScaleCUA model
# Model server should be running on port 10028

# Navigate to project root
cd "$(dirname "$0")/../../.."

# Activate virtual environment
source .venv/bin/activate

# Add osworld_agents and desktop_env to PYTHONPATH so they can be imported
export PYTHONPATH="${PYTHONPATH}:$(pwd)/rllm/agents:$(pwd)/rllm/environments/osworld"

# Configuration
# MODEL_NAME should be a valid HuggingFace model path for tokenizer loading
# For ScaleCUA, use "OpenGVLab/ScaleCUA-7B" which has unified tokenizer for text and vision
MODEL_NAME="OpenGVLab/ScaleCUA-3B"
MODEL_TYPE="qwen25vl"
API_URL="http://localhost:10028/v1"

# Environment settings
PROVIDER_NAME="vmware"  # Change to: virtualbox, docker, aws, etc. as needed
PATH_TO_VM=""           # Set path to your VM if using vmware/virtualbox
HEADLESS=""             # Add --headless flag if you want headless mode
ACTION_SPACE="pyautogui"
OBSERVATION_TYPE="a11y_tree"  # Options: screenshot, a11y_tree, screenshot_a11y_tree, som
SCREEN_WIDTH=1920
SCREEN_HEIGHT=1080

# Task settings
DOMAIN="all"  # Change to specific domain or "all" for all domains
TEST_CONFIG_BASE_DIR="examples/osworld/evaluation_examples"
TEST_ALL_META_PATH="examples/osworld/evaluation_examples/test_all.json"
RESULT_DIR="./results/osworld"

# Agent settings
MAX_STEPS=15
MAX_TRAJECTORY_LENGTH=3
HISTORY_N=5

# Model parameters
MAX_PIXELS=2109744
MIN_PIXELS=3136
TEMPERATURE=1.0
TOP_P=0.9
TOP_K=-1
MAX_TOKENS=500

# Parallelization
N_PARALLEL_AGENTS=1
BATCH_SIZE=1

# Run evaluation
python examples/osworld/run_osworld_agent.py \
    --model "$MODEL_NAME" \
    --model_type "$MODEL_TYPE" \
    --api_url "$API_URL" \
    --provider_name "$PROVIDER_NAME" \
    --action_space "$ACTION_SPACE" \
    --observation_type "$OBSERVATION_TYPE" \
    --screen_width "$SCREEN_WIDTH" \
    --screen_height "$SCREEN_HEIGHT" \
    --domain "$DOMAIN" \
    --test_config_base_dir "$TEST_CONFIG_BASE_DIR" \
    --test_all_meta_path "$TEST_ALL_META_PATH" \
    --result_dir "$RESULT_DIR" \
    --max_steps "$MAX_STEPS" \
    --max_trajectory_length "$MAX_TRAJECTORY_LENGTH" \
    --history_n "$HISTORY_N" \
    --max_pixels "$MAX_PIXELS" \
    --min_pixels "$MIN_PIXELS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --max_tokens "$MAX_TOKENS" \
    --n_parallel_agents "$N_PARALLEL_AGENTS" \
    --batch_size "$BATCH_SIZE" \
    $HEADLESS \
    ${PATH_TO_VM:+--path_to_vm "$PATH_TO_VM"}

