# ReflectVLM Repository Summary

## 1. Overview

This repository implements **ReflectVLM** (Reflective Planning: Vision-Language Models for Multi-Stage Long-Horizon Robotic Manipulation). It focuses on solving long-horizon robotic assembly tasks using a Vision-Language Model (VLM) that can "reflect" on its plans by imagining future states using a dynamics model (Simulator or Diffusion Model) and revising its actions.

## 2. Key Components

### Environment (`roboworld`)

- **Task**: Robotic Assembly (putting bricks into holes, reorienting, etc.).
- **Simulator**: MuJoCo.
- **Environment Class**: `FrankaAssemblyEnv` in `roboworld/envs/mujoco/franka/franka_assembly.py`.
- **Assets**: 3D models and XMLs in `roboworld/assets`.

### Agents (`roboworld/agent`)

- **LLaVA Agent** (`llava.py`): Wrapper around the LLaVA VLM to process images and prompts.
- **Oracle Agent** (`oracle.py`): A rule-based expert that uses symbolic states (READY, BLOCKED, etc.) to generate optimal actions. Used for data collection and as a reference.
- **Diffusion Simulator** (`diffuser.py`): A diffusion model used to imagine future frames given current image and action.

### Observation & Action Spaces

- **Observation Space**:
  - **Visual**: RGB images (current state, goal state).
  - **Textual**: Prompt describing the task, history of actions, and available objects.
- **Action Space**:
  - **Format**: Text strings.
  - **Primitives**: `pick up`, `put down`, `insert`, `reorient`, `done`.
  - **Arguments**: Object color/name (e.g., `pick up blue`).
  - **Example**: `pick up red`, `insert red`.

## 3. Baselines & Methods

### Base VLM (Baseline)

- **Script**: `scripts/eval_base_vlm.sh`
- **Method**: Uses a pre-trained LLaVA model to predict the next action directly from the current observation and history.
- **Command**: `bash scripts/eval_base_vlm.sh`

### ReflectVLM (Main Method)

- **Script**: `scripts/eval_reflect_vlm.sh`
- **Method**:
  1.  **Propose**: Base VLM proposes an initial plan (sequence of actions).
  2.  **Simulate**: A dynamics model (Simulator or Diffusion) generates a future image based on the first action of the plan.
  3.  **Reflect**: The VLM is prompted with the current state, the initial plan, and the _simulated future state_. It is asked to verify if the plan leads to a desirable future.
  4.  **Revise**: The VLM outputs a revised action if necessary.
- **Command**: `bash scripts/eval_reflect_vlm.sh {sim|diffusion}`

## 4. Evaluation

- **Script**: `run.py` is the main entry point.
- **Process**:
  - Loads the specified agent (LLaVA, Expert, etc.).
  - Runs `n_trajs` evaluation episodes.
  - In each step, queries the agent for an action, executes it in the MuJoCo environment, and logs the result.
- **Metrics**: Success Rate, Inference Time, Rollout Time.
- **Output**: Logs to WandB and saves a `meta.csv` with detailed trajectory info.

### Modifying Evaluation Config

You can modify flags in `scripts/eval_*.sh` or directly in `run.py`:

- `n_trajs`: Number of trajectories to evaluate.
- `level`: Difficulty level (`medium`, `hard`).
- `oracle_prob`: Probability of using the oracle action (for debugging/mixing).
- `agent_type`: `llava`, `expert`, or `random`.
- `revise_action`: Enable/disable reflection.
- `imagine_future_steps`: How many steps to look ahead.

## 5. Data Collection

- **Script**: `scripts/eval_expert.sh`
- **Method**: Runs the `OracleAgent` (`agent_type="expert"`) to generate optimal trajectories.
- **Output**:
  - Images: Saved in `logs/eval_expert/<traj_id>/`.
  - Metadata: `logs/eval_expert/meta.csv` containing history, actions, and file paths.

## 6. Training (Policy Learning)

- **Code**: `llava/train/` contains standard LLaVA training scripts.
- **Data Format**: To train the VLM, you need to convert the collected `meta.csv` and images into the LLaVA JSON format:
  ```json
  [
    {
      "id": "unique_id",
      "image": ["path/to/img1.png", "path/to/goal.png"],
      "conversations": [
        { "from": "human", "value": "<image>\n...Prompt..." },
        { "from": "gpt", "value": "pick up red" }
      ]
    }
  ]
  ```
- **Training Script**: `llava/train/train.py`.

## 7. Installation

The environment is managed via Conda (`reflectvlm`).

```bash
conda create -n reflectvlm python=3.9 -y
conda activate reflectvlm
pip install -e .
pip install -e ".[train]"  # For training support
```
