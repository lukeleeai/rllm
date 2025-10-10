# FrozenLake Agent Examples

This directory contains examples for training and running FrozenLake RL agents using the rLLM framework. The FrozenLake agent learns to navigate a grid world environment to reach a goal while avoiding holes.

Our examples demonstrate:
* Vision-language model support (Qwen2-VL)
* Visual observations (rendered grid images)
* Multimodal agent-environment interaction
* GRPO for training (currently text-only, VLM training coming soon)

## Environment Overview

FrozenLake is a classic reinforcement learning environment where:
- **Objective**: Navigate from start position to goal position
- **Observations**: Visual - rendered PNG images of the grid showing player (human icon), goal (treasure icon), holes (blue circles), and frozen tiles (white snow)
- **Actions**: Discrete movements - Up, Down, Left, Right
- **Dynamics**: Can be deterministic or slippery (stochastic movement)
- **Termination**: Episode ends when reaching goal (reward +1) or falling into hole (reward 0)
- **Parameters**:
  - `size`: Grid size (e.g., 4x4, 8x8)
  - `p`: Probability that the agent performs the intended action (remainder is split among unintended directions)
  - `seed`: Random seed for environment generation
  - `is_slippery`: Boolean flag controlling whether movement is stochastic (True) or deterministic (False)

## Architecture

### Component Roles

**Environment (FrozenLakeEnv)**
- Simulates the frozen lake game world
- Generates visual observations as base64-encoded PNG images
- Accepts discrete actions (1=Left, 2=Down, 3=Right, 4=Up)
- Returns: observation image, reward, done flag, info dict

**Agent (FrozenLakeAgent)**
- Maintains conversation history in multimodal format
- Accumulates full message history: system prompt, user messages with images, assistant responses
- Parses actions from model responses using regex matching
- Does NOT communicate with the model directly

**Execution Engine (AgentExecutionEngine)**
- Orchestrates the agent-environment interaction loop
- Sends messages to the vision-language model (line 322 in agent_execution_engine.py)
- Manages tokenization and training data preparation
- Accumulates prompt_tokens (initial context) and response_tokens (full trajectory)

### Data Flow

For inference:
1. Environment generates PNG image observation
2. Agent wraps image in multimodal message format
3. Engine sends full conversation history (with all images) to VLM model
4. Model returns response with action
5. Agent parses action
6. Environment executes action and generates new image
7. Repeat until episode ends

For training (CURRENT LIMITATION):
- Text tokens are extracted and accumulated correctly
- Visual tokens are NOT yet supported (images stripped during tokenization)
- TODO: Add processor support for proper VLM training with visual inputs

## Model Hosting

### Using vLLM with Vision-Language Models

Start a vLLM server with OpenAI-compatible API for vision models:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model unsloth/Qwen2-VL-7B-Instruct \
    --host 0.0.0.0 \
    --port 30000 \
    --max-model-len 16384 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.8
```

Note: Vision-language models like Qwen2-VL require additional memory. Adjust max-model-len based on your GPU capacity.

The server should be accessible at `http://localhost:30000/v1`

## Dataset Preparation

Prepare the FrozenLake datasets (randomly generated environments for training and testing):

```bash
cd examples/frozenlake
python prepare_frozenlake_data.py
```

This will:
- Generate 10,000 random FrozenLake environments for training
- Generate 100 random FrozenLake environments for testing
- Register both datasets with the RLLM DatasetRegistry
- Each environment has random size (2-10), slip probability (0.6-0.85), and seed

## Running Inference

Once your model server is running and datasets are prepared, you can run inference:

```bash
cd examples/frozenlake
python run_frozenlake_agent.py
```

### Configuration Options

You can modify the inference script parameters:

- `n_parallel_agents`: Number of parallel agents (default: 1 for debugging)
- `model_name`: Model to use (default: "unsloth/Qwen2-VL-7B-Instruct")
- `base_url`: API server URL (default: "http://localhost:30000/v1")
- `max_response_length`: Maximum response length (default: 2048)
- `max_prompt_length`: Maximum prompt length (default: 2048)
- `max_steps`: Maximum steps per episode (default: 20)
- `temperature`: Sampling temperature (default: 0.6)
- `top_p`: Top-p sampling (default: 0.95)

The script will:
1. Load the FrozenLake test dataset (or generate if not exists)
2. Create agents and environments
3. Run inference through the async agent execution engine
4. Send multimodal messages (images + text) to the VLM model
5. Evaluate results and compute pass@k metrics

### Current Inference Capabilities

- Fully functional multimodal inference with vision-language models
- Images are correctly accumulated in conversation history
- All images from previous steps are sent to the model as context
- Model can see the full visual trajectory to make informed decisions

## Training

### Current Status

Training infrastructure is currently being updated to support vision-language models:

**What Works:**
- Text-based token accumulation and masking
- Trajectory collection with rewards and MC returns
- GRPO/PPO training for text-only models

**What Needs Implementation (see TODOs in code):**
- Processor support for multimodal inputs
- Visual token handling (image placeholder tokens)
- Pixel values and image metadata storage
- Vision encoder integration in training loop

### Text-Only Training (Current)

To train a text-based FrozenLake agent:

```bash
bash examples/frozenlake/train_frozenlake_agent.sh
```

Note: This uses text grid representations instead of images. For full vision-language model training with visual observations, the processor infrastructure needs to be implemented first.

### Future: Vision-Language Model Training

Once the VLM training infrastructure is complete, the training pipeline will:
1. Use processor to tokenize text and encode images
2. Insert special image token placeholders in token sequences
3. Store pixel_values and image_grid_thw alongside tokens
4. Pass both text tokens and visual data to the model during training
5. Compute gradients over both textual and visual inputs
