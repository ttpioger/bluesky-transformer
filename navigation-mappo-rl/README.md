# Multi Agent Navigation

![Multi Agent Navigation Demo](images/marl1.gif)

## Project Description
The core RL task involves multiple agents (represented as colored circles and arrow-heads) navigating from their starting positions to their respective end goals while avoiding collisions with other agents and static obstacles. 

To do this, we are using a CTDE (Centralized Training Decentralized Execution) approach to train agents. Specifically, we are using MAPPO (Multi-Agent Proximal Policy Optimization), a variant of PPO inspired by MADDPG.


# Video Tutorial

> **📺 Watch the Video for free**  
> **[Training RL Agents to Navigate as a Team](https://youtu.be/Ji5VTbH7i08)**  

## Support

If you find this content helpful, please consider supporting my work on Patreon. Your support helps me create more in-depth tutorials and content. 

Patreon members also get access to a code walkthrough of this repo.

[<img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patron!" width="200">](https://www.patreon.com/NeuralBreakdownwithAVB)

## Getting Started

### Prerequisites

-   **Python 3.10+** (required)
-   **`uv`** (recommended) or `pip` for package management
-   To render movies and videos, install ffmpeg

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/avbiswas/navigation-mappo-rl.git
    ```

2.  **Install dependencies:**
    ```bash
    # Using uv
    uv sync
    ```

3.  **Train a new model:**
    ```bash
    uv run train_mappo.py model_id configs/config.yaml
    ```

    For example:
    ```bash
    uv run train_mappo.py model_1 configs/basic_env.yaml
    ```

    This will create a new model inside `models/model_1`. The latest model and the best all-time models are both saved.

4. **Run inference**
    ```bash
    uv run inference.py models/model_1
    ```

    By default, this will test on the environment config file where the model was originally trained. You can also test on a new config though.

    ```bash
    uv run inference.py models/model_1 configs/bottleneck.yaml
    ```

## Included Environments

A variety of pre-configured environments are provided for your experiments in the `configs/` directory.

1. Basic

   ![Basic Environment](images/basic.png)

2. Circle

   ![Circle Environment](images/circle.png)

3. Moving Environment

   ![Moving Environment](images/moving.png)

4. Hallway

   ![Hallway](images/hallway.png)

5. Bottleneck

   ![Bottleneck](images/bottleneck.png)

6. Four Crossing

   ![Four Crossing](images/fourcross.png)

## Creating Custom Environments

You can define your own environments by creating a new `.yaml` file in the `configs/` directory. The configuration schema is defined in `nav/config_models.py`.

Key components of a configuration file:
- **Boundary**: Defines the polygon vertices for the playable area.
- **Agents**: Specifies start/goal zones (rectangles), physical properties (radius, max speed), and sensor settings (FOV, range).
- **Obstacles**: Defines static or moving obstacles (rectangles or circles).

For reference, check `configs/basic_env.yaml` for a simple setup or `configs/moving_env.yaml` for dynamic obstacles.

## MAPPO Configuration

The training configuration and hyperparameters are defined in `train_mappo.py`.

Key settings include:
- **History Length**: `history_length = 4` (Number of past frames stacked).
- **Batch Size**: `batch_size=128` (Number of samples per update).
- **Learning Rate**: `learning_rate=5e-4`.
- **Inference Interval**: `inference_interval=5` (How often to run evaluation episodes).
- **Network Architecture**: The policy network uses an `ObservationEncoder` which processes LIDAR data and agent states, outputting a feature vector of size 384.

To modify these, edit the `MAPPO` initialization in `train_mappo.py`.

## Rendering

Check out `nav/live_renderer.py` to see useful rendering settings.

## Environment Details

The environment specifications are defined in `nav/environment.py`.

### Observation Space (`Box`)
Each agent receives a composite observation consisting of:
1.  **State Vector**:
    -   Progress towards goal (normalized 0-1).
    -   Cosine of the angle between the agent's heading and the goal.
    -   Current speed ratio (current_speed / max_speed).
    -   Distance to goal.
    -   Goal vector (x, y).
2.  **LIDAR Readings**:
    -   A set of raycasts (default 60 rays).
    -   Each ray returns 3 channels: [Distance to Obstacle, Distance to Boundary, Distance to Agent].
    -   Stacked with `history_length` (default 4) frames to provide temporal context.

### Action Space (`Box(2,)`)
The agent controls its movement via a continuous 2D vector:
-   `[vx, vy]`: Velocity components in the X and Y directions.
-   Values are clipped to the range `[-1, 1]` and scaled by the agent's `max_speed`.
-   The action is applied in the Local Coordinate Space of the agent, where the origin is at the agent's center, and the Y-axis is the agent's goal vector.

### Reward Structure
The reward function incentivizes reaching the goal while avoiding collisions:
-   **Goal Reached**: `+10`
-   **Collision** (Obstacle, Boundary, or Agent): `-10`
-   **Progress Reward**: Scaled by speed and alignment with the goal direction (encourages moving efficiently towards the target).
-   **Time Penalty (Stay Alive)**: `-0.05` per step (encourages reaching the goal quickly).


## Network Architecture

The model architecture (found in `rl/mappo.py` and `networks/actor_critic_network.py`) implements the Actor-Critic method with shared features.

### Shared Observation Encoder (`ObservationEncoder`)
Both the Actor and Critic networks share a common backbone called the `ObservationEncoder`. This ensures that feature extraction from the raw environment state is consistent and efficient.
- **Input**: Takes the agent's observation, which includes stacked history frames.
- **Structure**: 
    - **1D CNN**: Processes the LIDAR data (which is treated as a 1D sequence of ray intersections).
    - **Agent Network (MLP)**: Processes the scalar agent states (velocity, distance to goal, etc.).
    - **Dense Layer**: Concatenates the outputs of the CNN and the MLP to produce a unified feature vector (default dimension 384).

### Centralized Critic (`CentralizedCriticNetwork`)
The critic evaluates the value of a state for **all agents**, making it "centralized". 
- **Attention Mechanism**: It uses a Multi-Head Attention (MHA) layer to process the states of all agents. By attending to other agents' encoded states, the critic can learn cooperative behaviors and anticipate interactions.
- **Output**: A single scalar value representing the estimated return (value) of the global state.

Currently the environment also has a "state" function, but we are not using it, left for future work.


### Decentralized Actor (`DecentralizedActorNetwork`)
The actor decides the action for a **single agent**, making it "decentralized" during execution.
- **Input**: Takes the shared feature vector from the `ObservationEncoder`.
- **Structure**: A standard MLP (Multi-Layer Perceptron) that maps features to action probabilities.
- **Output**: A probability distribution (Diagonal Gaussian) over the action space, from which the agent samples its movement commands.
