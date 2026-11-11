import pettingzoo
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .obstacles import PolygonBoundary
from .utils import sample_point_in_rectangle, convert_to_polar
from .config_models import *
from .renderer_models import RenderState, AgentState, BoundaryState
from .obstacles import ObstacleFactory
import yaml
from typing import Dict, List, Any, Literal, Optional
from pydantic import BaseModel
from .ray_intersection import (
    batch_ray_intersection_detailed,
    create_lidar_rays,
    RayIntersectionOutput,
)
from collections import deque
from .live_renderer import SimulationWindow

DELTA_T = 1 / 60

COLLIDING_WITH_TYPES = Literal["obstacle", "boundary", "agent"]


class CollisionData(BaseModel):
    is_colliding: bool = False
    colliding_with: Optional[COLLIDING_WITH_TYPES] = None


class Agent:
    def __init__(
        self,
        agent_config: AgentConfig,
        goal_threshold: float = 0.02,
    ):
        self.config = agent_config
        self.pos = self.config.start_pos.center.to_numpy()
        self.start_pos = self.pos.copy()
        self.radius = self.config.radius
        self.current_speed = 0.1
        self.goal_sample_area = self.config.goal_pos
        self.goal_pos = self.config.goal_pos.center.to_numpy()
        _, self.direction = convert_to_polar(self.goal_pos - self.pos)
        self.response_time = 10
        self.active = True
        self.lidar_observation_history = deque(maxlen=4)
        self.last_raw_lidar_observation = None
        self.goal_reached = False
        self.old_pos = self.pos.copy()
        self.goal_threshold = goal_threshold
        self.last_reward = 0

    def has_reached_goal(self):
        return np.linalg.norm(self.goal_pos - self.pos) < (
            self.goal_threshold + self.radius
        )

    def get_state_dict(self):
        original_distance_to_goal = np.linalg.norm(self.goal_pos - self.start_pos)
        current_distance_to_goal = np.linalg.norm(self.goal_pos - self.pos)
        progress = (
            original_distance_to_goal - current_distance_to_goal
        ) / original_distance_to_goal

        goal_vector = self.goal_pos - self.pos
        goal_vector = goal_vector / np.linalg.norm(goal_vector)
        cosine_angle = goal_vector.dot(self.direction)

        speed_ratio = self.current_speed / self.config.max_speed

        return {
            "state_vector": [
                progress,  # progress towards goal 0-1
                cosine_angle,  # cosine of angle between goal vector and direction vector
                speed_ratio,  # ratio of current speed to max speed
                speed_ratio * cosine_angle,
                current_distance_to_goal,
                goal_vector[0],
                goal_vector[1],
            ],
        }

    def get_action(self, lidar_observation: np.ndarray):
        self.lidar_observation_history.append(lidar_observation)
        return None  # TODO: Implement action selection

    def update_pos(self, delta_t: float = 1 / 30):

        if self.goal_reached:
            return

        if not self.active:
            return
        self.old_pos = self.pos.copy()
        self.pos = self.pos + (self.direction * self.current_speed * delta_t)
        self.goal_reached = self.has_reached_goal()

    def convert_velocity_to_global(self, velocity: Vector2, heading_vector: Vector2):
        dist = np.linalg.norm(heading_vector)
        if dist > 1e-10:
            # Normalized direction to goal (New Y-axis)
            u_x = heading_vector[0] / dist
            u_y = heading_vector[1] / dist

            # Action components
            a_x = velocity.x
            a_y = velocity.y

            # Transform to global: v_global = a_x * Right + a_y * Forward
            # Right vector corresponds to (u_y, -u_x)
            global_vx = a_x * u_y + a_y * u_x
            global_vy = -a_x * u_x + a_y * u_y

            target_velocity_global = np.array([global_vx, global_vy])
        else:
            # If already at goal, keep existing behavior or zero out
            target_velocity_global = velocity.to_numpy()

        return target_velocity_global

    def apply_target_velocity(self, target_velocity: Vector2):
        if not self.active:
            self.current_speed = 0
            return

        target_velocity_global = self.convert_velocity_to_global(
            target_velocity, (self.goal_pos - self.pos)

        )

        current_velocity = self.current_speed * self.direction
        force = target_velocity_global - current_velocity
        new_velocity = current_velocity + force * (self.response_time * DELTA_T)

        # Handle zero velocity case to avoid division by zero
        velocity_magnitude = np.linalg.norm(new_velocity)
        if velocity_magnitude > 1e-10:  # Small threshold to avoid numerical issues
            self.current_speed = velocity_magnitude
            self.direction = (
                new_velocity / velocity_magnitude
            )  # Normalize to unit vector
        else:
            self.current_speed = 0.0
            # Keep previous direction when speed is zero

        # Clamp speed to maximum (was incorrectly using max instead of min)
        self.current_speed = min(self.current_speed, self.config.max_speed)


class Environment(pettingzoo.ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array", "none"], "name": "navigation_v0"}

    def __init__(
        self,
        config: dict[str, Any] | EnvConfig,
        render_mode: Optional[str] = None,
        avoid_collision_checks: bool = False,
    ):
        if isinstance(config, dict):
            config = EnvConfig(**config)

        self.config = config
        self.state_image_size = config.state_image_size
        self.boundary = PolygonBoundary(config.boundary)
        self.num_groups = len(config.agents)  # rename config.agents to config.groups
        self.avoid_collision_checks = avoid_collision_checks

        agent_configs = self.preprocess_agent_configs(
            config.agents, config.num_agents_per_group
        )

        self.agents_dict: Dict[str, Agent] = {
            f"agent_{i}": Agent(
                agent_config,
                goal_threshold=self.config.goal_threshold,
            )
            for i, agent_config in enumerate(agent_configs)
        }

        self.n_agents = len(self.agents_dict)
        self.state_dim = len(
            next(iter(self.agents_dict.values())).get_state_dict()["state_vector"]
        )
        self.obstacles = [
            ObstacleFactory.create(obstacle) for obstacle in config.obstacles
        ]
        self.num_steps = 0

        # PettingZoo required attributes
        self.possible_agents = list(self.agents_dict.keys())
        self.agents = self.possible_agents.copy()

        # Define observation and action spaces
        self._setup_spaces()

        # Rendering setup
        self.render_mode = render_mode
        self.window = None
        if render_mode == "human" or render_mode == "rgb_array":
            # Use headless mode for rgb_array to avoid showing window
            headless = render_mode == "rgb_array"
            self.window = SimulationWindow(
                target_fps=30, record=True, headless=headless
            )

    def get_window(self):
        return self.window

    def preprocess_agent_configs(
        self, agent_configs: List[AgentConfig], num_agents_per_group: int
    ):
        """Preprocess agent configs to create multiple agents per group."""
        processed_configs = []
        num_groups = len(agent_configs)

        for group_idx, agent_config in enumerate(agent_configs):
            start_rect = agent_config.start_pos
            width = start_rect.width
            middle = start_rect.center.x
            spacing = width / num_agents_per_group

            for i in range(num_agents_per_group):
                new_agent_config = agent_config.model_copy()

                if i == 0:
                    x_offset = 0
                elif i % 2 == 1:
                    x_offset = -((i + 1) // 2) * spacing
                else:
                    x_offset = (i // 2) * spacing

                this_center = Vector2(
                    x=middle + x_offset,
                    y=start_rect.center.y,
                )
                new_rect = Rectangle(
                    center=this_center,
                    width=(width / num_agents_per_group - new_agent_config.radius * 2),
                    height=start_rect.height,
                )
                new_agent_config.start_pos = new_rect

                processed_configs.append(new_agent_config)
        return processed_configs

    def _setup_spaces(self):
        """Setup observation and action spaces for all agents."""
        # Action space: 2D velocity vector (vx, vy)
        max_speed = max(agent.config.max_speed for agent in self.agents_dict.values())
        self._action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Observation space: state vector + lidar readings
        # State vector: [progress, cosine_angle, speed_ratio, speed_ratio * cosine_angle, distance_to_goal]
        state_dim = self.agent_states_dim
        lidar_dim = (
            self.config.num_rays * 3
        )  # 3 channels per ray (obstacle, boundary, agent)

        obs_low = np.concatenate(
            [
                np.array([-1] * state_dim),  # state vector bounds
                np.zeros(lidar_dim),  # lidar readings are non-negative
            ]
        )
        obs_high = np.concatenate(
            [
                np.array(
                    [1] * state_dim
                ),  # state vector bounds, externally guaranteed distance max = 1
                np.full(lidar_dim, max_speed),  # max lidar range
            ]
        )

        self._observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

    @property
    def agent_states_dim(self):
        return self.state_dim

    @property
    def lidar_dim(self):
        return self.config.num_rays * 3

    @property
    def observation_spaces(self):
        """Returns observation spaces for all agents."""
        return {agent: self._observation_space for agent in self.agents}

    @property
    def action_spaces(self):
        """Returns action spaces for all agents."""
        return {agent: self._action_space for agent in self.agents}

    def observation_space(self, agent):
        """Returns observation space for a specific agent."""
        return self._observation_space

    def action_space(self, agent):
        """Returns action space for a specific agent."""
        return self._action_space

    def is_point_colliding(
        self, pos: np.ndarray, starting_points: List[np.ndarray], radius: float
    ) -> bool:
        for starting_point in starting_points:
            if np.linalg.norm(pos - starting_point) < radius:
                return True
        return False

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        if seed is not None:
            np.random.seed(seed)

        starting_points = []

        # Reset all agents to initial positions
        for agent in self.agents_dict.values():

            pos = sample_point_in_rectangle(agent.config.start_pos)

            num_tries = 0

            # Checking for collision on frame 1 during init
            while (
                self.is_point_colliding(pos, starting_points, agent.radius*2.25)
                and num_tries < 10
            ):
                pos = sample_point_in_rectangle(agent.config.start_pos)
                num_tries += 1

            starting_points.append(pos)
            agent.pos = pos

            agent.goal_pos = sample_point_in_rectangle(agent.goal_sample_area)
            agent.current_speed = 0.1
            _, agent.direction = convert_to_polar(agent.goal_pos - agent.pos)
            agent.active = True
            agent.goal_reached = False
            agent.lidar_observation_history.clear()
            agent.last_raw_lidar_observation = None
            agent.old_pos = agent.pos.copy()

        # Reset obstacles
        for obs in self.obstacles:
            if hasattr(obs, "reset"):
                obs.reset()

        self.num_steps = 0
        self.agents = self.possible_agents.copy()
        self.num_dead_agents = 0

        # Get initial observations
        observations = self._get_observations()
        infos = {agent: {} for agent in self.agents}

        # Render initial state if render mode is set
        if self.render_mode == "human":
            self.render()

        return observations, infos

    def _get_observations(self):
        """Get observations for all agents."""
        lidar_observations = self.get_lidar_observation()

        # Update agent lidar data
        for agent_id, lidar_observation in zip(self.agents, lidar_observations):
            self.agents_dict[agent_id].last_raw_lidar_observation = lidar_observation

        processed_lidar_observations = [
            self.process_lidar_observation(
                self.agents_dict[agent_id].config.max_range, lidar_observation
            )
            for agent_id, lidar_observation in zip(self.agents, lidar_observations)
        ]

        observations = {}
        for agent_idx, agent_id in enumerate(self.agents):
            agent = self.agents_dict[agent_id]
            state_dict = agent.get_state_dict()
            state_vector = np.array(state_dict["state_vector"], dtype=np.float32)
            lidar_vector = (
                processed_lidar_observations[agent_idx].flatten().astype(np.float32)
            )

            # concat the state features and the lidar features together
            observations[agent_id] = np.concatenate([state_vector, lidar_vector])

        return observations

    def calculate_reward(self, agent: Agent, collision_data: CollisionData):
        if agent.goal_reached:
            return 10

        if collision_data.is_colliding:
            return -10

        goal_reward = agent.direction.dot(agent.goal_pos - agent.pos)
        scale_goal_reward_with_speed = goal_reward * (
            agent.current_speed / agent.config.max_speed
        )

        stay_alive_reward = -0.05
        scale_goal_reward_with_speed *= 0.25

        return scale_goal_reward_with_speed + stay_alive_reward

    def state(self):
        """
        Return occupancy grids for the agents, the goals, and the obstacles
        """

        # Currently not used to train the central critic, but this returns a full fledged state vector
        
        grid_size = self.config.state_image_size
        agent_occupancy_grid = np.zeros([self.n_agents, grid_size, grid_size])
        goal_occupancy_grid = np.zeros([self.n_agents, grid_size, grid_size])
        obstacle_occupancy_grid = np.zeros([1, grid_size, grid_size])

        for idx, (_, agent) in enumerate(self.agents_dict.items()):
            start_pos = agent.pos
            goal_pos = agent.goal_pos

            agent_x_grid = int(
                np.clip(start_pos[0] * grid_size, 0, grid_size - 1)
            )  # Column (X)
            agent_y_world_grid = int(
                np.clip(start_pos[1] * grid_size, 0, grid_size - 1)
            )
            agent_y_grid = grid_size - 1 - agent_y_world_grid  # Row (Y), flipped

            goal_x_grid = int(
                np.clip(goal_pos[0] * grid_size, 0, grid_size - 1)
            )  # Column (X)
            goal_y_world_grid = int(np.clip(goal_pos[1] * grid_size, 0, grid_size - 1))
            goal_y_grid = grid_size - 1 - goal_y_world_grid  # Row (Y), flipped

            # Store as [agent_idx, row, column] = [agent_idx, y, x]
            agent_occupancy_grid[idx, agent_y_grid, agent_x_grid] = 1
            goal_occupancy_grid[idx, goal_y_grid, goal_x_grid] = 1

        cell_size = 1.0 / grid_size  # Size of each cell in world coordinates
        for i in range(grid_size):
            for j in range(grid_size):
                world_x = (j + 0.5) * cell_size  # j is column, maps to x
                world_y = (grid_size - i - 0.5) * cell_size  # Flip i to get correct y
                world_coord = np.array([world_x, world_y])

                for obs in self.obstacles:
                    if obs.check_collision(center=world_coord, radius=cell_size * 0.5):
                        obstacle_occupancy_grid[0, i, j] = 1

        return {
            "agent_occupancy": agent_occupancy_grid,
            "goal_occupancy": goal_occupancy_grid,
            "obstacle_occupancy": obstacle_occupancy_grid,
        }

    def transition(self, actions: list[Vector2]):

        terminations = {}
        truncations = {}
        collision_datas: list[CollisionData] = []

        # Apply actions to agents
        for agent_id, action in zip(self.agents, actions):
            agent = self.agents_dict[agent_id]
            target_velocity = Vector2(
                x=action.x * agent.config.max_speed,
                y=action.y * agent.config.max_speed,
            )
            agent.apply_target_velocity(target_velocity)

        # Update obstacles
        for obs in self.obstacles:
            obs.update(DELTA_T)

        # Update agent positions and check collisions
        collision_datas = []
        for agent_id in self.agents:
            agent = self.agents_dict[agent_id]
            agent.update_pos(DELTA_T)

            if self.avoid_collision_checks:
                continue

            this_agent_collision_data = CollisionData()
            
            # What's colliding - boundaries, obstacles, or agents?
            if self.boundary.violating_boundary(agent):
                agent.active = False
                this_agent_collision_data.is_colliding = True
                this_agent_collision_data.colliding_with = "boundary"

            for obs in self.obstacles:
                if obs.check_collision(center=agent.pos, radius=agent.radius):
                    agent.active = False
                    this_agent_collision_data.is_colliding = True
                    this_agent_collision_data.colliding_with = "obstacle"

            # TODO: Check for collisions with other agents
            for other_agent_id in self.agents:
                if other_agent_id == agent_id:
                    continue
                other_agent = self.agents_dict[other_agent_id]
                if np.linalg.norm(other_agent.pos - agent.pos) < (
                    agent.radius + other_agent.radius
                ):
                    agent.active = False
                    this_agent_collision_data.is_colliding = True
                    this_agent_collision_data.colliding_with = "agent"

            collision_datas.append(this_agent_collision_data)

        for agent_id in self.agents:
            agent = self.agents_dict[agent_id]

            terminations[agent_id] = (not agent.active) or agent.goal_reached
            truncations[agent_id] = (
                False if self.num_steps < self.config.max_time else True
            )

            # a pettingzoo quirk where terminations also need to be truncated
            terminations[agent_id] = np.bool_(
                terminations[agent_id] or truncations[agent_id]
            )

        return collision_datas, terminations, truncations

    def step(self, actions: dict[str, np.ndarray]):
        """Execute one step of the environment."""
        # Handle both dictionary format (original) and vectorized format (after SuperSuit wrapping)
        if isinstance(actions, dict):
            # Original PettingZoo format: {"agent_0": [vx, vy]}
            processed_actions = []
            for agent_id in self.agents:
                if agent_id in actions:
                    action = actions[agent_id]
                    processed_actions.append(
                        Vector2(x=float(action[0]), y=float(action[1]))
                    )
                else:
                    processed_actions.append(Vector2(x=0.0, y=0.0))

        self.num_steps += 1

        # Repeat the chosen action for multiple frames
        # This avoids re-calculation of nearby frames (inspired by Atari DQN)
        for _ in range(self.config.repeat_steps):
            collision_datas, terminations, truncations = self.transition(
                processed_actions
            )

            if any(terminations.values()):
                break

            if any(truncations.values()):
                break

        # Calculate rewards
        rewards = {}
        for agent_id, collision_data in zip(self.agents, collision_datas):
            agent = self.agents_dict[agent_id]
            rewards[agent_id] = self.calculate_reward(agent, collision_data)
            if collision_data.is_colliding:
                self.num_dead_agents += 1

        for agent_id in self.agents_dict.keys():
            agent = self.agents_dict[agent_id]
            if agent_id in rewards:
                agent.last_reward = rewards[agent_id]
        # Get next observations
        observations = self._get_observations()
        infos = {agent: {} for agent in self.agents}

        # Remove terminated agents
        self.agents = [
            agent_id for agent_id in self.agents if not terminations[agent_id]
        ]

        # Auto-render if render mode is set to human
        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def close(self):
        """Close the environment."""
        if self.window is not None:
            self.window.on_close()
            self.window.close()
            self.window = None

    def render(self):
        render_state = self.get_render_state()
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self.window.render(render_state)
            return self.window.get_rgb_array()
        else:
            return render_state

    def process_lidar_observation(
        self, max_range, lidar_observation: list[RayIntersectionOutput]
    ):
        rays = np.zeros((3, len(lidar_observation)))
        for i in range(len(lidar_observation)):
            ray_data = lidar_observation[i]
            if not ray_data.intersects:
                continue

            if ray_data.intersecting_with == "obstacle":
                rays[0, i] = max_range - ray_data.t
            elif ray_data.intersecting_with == "boundary":
                rays[1, i] = max_range - ray_data.t
            elif ray_data.intersecting_with == "agent":
                rays[2, i] = max_range - ray_data.t

        return rays

    def get_lidar_observation(self):
        all_rays = []
        goals = []
        rays_per_agent = []

        for agent_id in self.agents:
            agent = self.agents_dict[agent_id]
            rays = create_lidar_rays(
                agent.pos,
                agent.direction,
                self.config.num_rays,
                agent.config.max_range,
                agent.config.fov_degrees,
            )
            all_rays.extend(rays)

            goals.append(
                Circle(
                    center=Vector2(x=agent.goal_pos[0], y=agent.goal_pos[1]),
                    radius=self.config.goal_threshold,
                )
            )
            rays_per_agent.append(self.config.num_rays)

        if not all_rays:
            return []

        all_rays = np.array(all_rays)
        result = batch_ray_intersection_detailed(
            all_rays,
            self.obstacles,
            [self.config.boundary],
            goals=goals,
            agents=[
                Circle(
                    center=Vector2(
                        x=self.agents_dict[agent].pos[0],
                        y=self.agents_dict[agent].pos[1],
                    ),
                    radius=self.agents_dict[agent].radius,
                )
                for agent in self.agents
            ],
            rays_per_agent=rays_per_agent,
        )

        result = np.reshape(result, (len(self.agents), self.config.num_rays))

        return result

    def get_render_state(self) -> RenderState:
        agent_states = []
        for agent_id in self.agents:
            agent = self.agents_dict[agent_id]
            agent_states.append(
                AgentState(
                    position=(agent.pos[0], agent.pos[1]),
                    radius=agent.radius,
                    color=agent.config.agent_col,
                    velocity=(
                        agent.current_speed * agent.direction[0],
                        agent.current_speed * agent.direction[1],
                    ),
                    direction=(agent.direction[0], agent.direction[1]),
                    lidar_observation=agent.last_raw_lidar_observation,
                    fov_degrees=agent.config.fov_degrees,
                    max_range=agent.config.max_range,
                    goals=Circle(
                        center=Vector2(x=agent.goal_pos[0], y=agent.goal_pos[1]),
                        radius=self.config.goal_threshold,
                    ),
                    goal_reached=agent.goal_reached,
                    last_reward=agent.last_reward,
                )
            )

        obstacle_states = []
        for obs in self.obstacles:
            obstacle_states.append(obs.get_current_state())

        boundary_state = BoundaryState(
            vertices=[(v[0], v[1]) for v in self.boundary.vertices]
        )
        
        # Everything live_renderer needs to display the current frame
        return RenderState(
            agents=agent_states,
            obstacles=obstacle_states,
            boundary=boundary_state,
        )

