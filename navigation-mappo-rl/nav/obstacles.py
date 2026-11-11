import numpy as np
from abc import ABC, abstractmethod
from typing import Union

from .config_models import *
from .utils import *

OBSTACLE_NOISE = 0.02


def get_random_noise(noise=OBSTACLE_NOISE) -> float:
    return np.random.uniform(-noise, noise)


class Obstacle(ABC):
    def __init__(self, config: ObstacleConfig):
        self.config = config
        self.schedule = config.schedule

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def check_collision(self, center, radius) -> bool:
        pass

    @abstractmethod
    def update(self, delta_t):
        pass

    @abstractmethod
    def get_current_state(self) -> Union[Rectangle, Circle]:
        pass


class ObstacleFactory:
    @classmethod
    def create(cls, config: ObstacleConfig) -> Obstacle:
        if isinstance(config.shape, Rectangle):
            return RectangleObstacle(config)
        elif isinstance(config.shape, Circle):
            return CircleObstacle(config)


class RectangleObstacle(Obstacle):
    def __init__(self, config: ObstacleConfig):
        super().__init__(config)
        shape = config.shape
        if not isinstance(shape, Rectangle):
            return
        self.config = config
        self.center = shape.center.to_numpy()
        self.width = shape.width
        self.height = shape.height
        self.rotation = shape.rotation
        self.noise = config.noise if config.noise is not None else OBSTACLE_NOISE

    def reset(self):
        shape = self.config.shape
        self.center = shape.center.to_numpy() + np.array(
            [get_random_noise(self.noise), get_random_noise(self.noise)]
        )
        self.width = shape.width + get_random_noise(self.noise)
        self.height = shape.height + get_random_noise(self.noise)
        if self.noise == 0:
            self.rotation = 0
        else:
            self.rotation = shape.rotation + np.random.uniform(-10, 10)

    def get_current_state(self):
        return Rectangle(
            center=Vector2(x=self.center[0], y=self.center[1]),
            width=self.width,
            height=self.height,
            rotation=self.rotation,
        )

    def update(self, delta_t):
        if self.schedule is None:
            return

        if self.schedule.direction is not None and self.schedule.speed is not None:
            self.center += (
                (self.schedule.speed + get_random_noise())
                * self.schedule.direction.to_numpy()
                * delta_t
            )

        if self.schedule.angular_speed is not None:
            # Angular speed is in degrees per second, convert to degrees per frame
            self.rotation += (
                self.schedule.angular_speed + get_random_noise(self.noise)
            ) * delta_t

        self._handle_boundary_conditions()

    def _handle_boundary_conditions(self):
        if self.schedule is None:
            return

        x_min = self.schedule.boundary_x_min
        x_max = self.schedule.boundary_x_max

        if x_min is not None and self.center[0] < x_min:
            if self.schedule.direction is not None:
                self.schedule.direction.x *= -1

        if x_max is not None and self.center[0] > x_max:
            if self.schedule.direction is not None:
                self.schedule.direction.x *= -1

    def check_collision(self, center, radius) -> bool:
        # Translate and rotate the agent's center to the rectangle's local coordinates
        agent_center_local = center - self.center

        # Convert rotation from degrees to radians for trigonometric functions
        theta = -np.radians(self.rotation)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        rotated_x = (
            agent_center_local[0] * cos_theta - agent_center_local[1] * sin_theta
        )
        rotated_y = (
            agent_center_local[0] * sin_theta + agent_center_local[1] * cos_theta
        )

        rotated_agent_center = np.array([rotated_x, rotated_y])

        # Find the closest point on the AABB to the rotated agent's center
        half_width = self.width / 2
        half_height = self.height / 2

        # Vector from center of AABB to rotated agent center
        v = rotated_agent_center

        # Vector from agent center to closest point on AABB
        u = np.maximum(np.abs(v) - np.array([half_width, half_height]), 0)

        # Squared distance
        distance_sq = np.sum(u**2)

        # If this distance is less than the circle's radius, a collision occurs
        collided = bool(distance_sq <= radius**2)
        return collided


class CircleObstacle(Obstacle):
    def __init__(self, config: ObstacleConfig):
        super().__init__(config)
        shape_config = config.shape
        if not isinstance(shape_config, Circle):
            return
        self.center = shape_config.center.to_numpy()
        self.radius = shape_config.radius
        self.noise = config.noise if config.noise is not None else OBSTACLE_NOISE

    def reset(self):
        shape = self.config.shape
        self.center = shape.center.to_numpy() + np.array(
            [get_random_noise(self.noise), get_random_noise(self.noise)]
        )
        self.radius = shape.radius + get_random_noise(self.noise)

    def get_current_state(self):
        return Circle(
            center=Vector2(x=self.center[0], y=self.center[1]), radius=self.radius
        )

    def update(self, delta_t):
        if self.schedule is None:
            return
        if self.schedule.speed is not None and self.schedule.direction is not None:
            self.center += (
                self.schedule.direction.to_numpy()
                * (self.schedule.speed + get_random_noise(self.noise))
                * delta_t
            )
        self._handle_boundary_conditions()

    def _handle_boundary_conditions(self):
        if self.schedule is None:
            return
        x_min = self.schedule.boundary_x_min
        x_max = self.schedule.boundary_x_max

        if x_min is not None and self.center[0] < x_min:
            if self.schedule.on_boundary == "reset":
                self._reset_state()
            elif self.schedule.on_boundary == "oscillate":
                self.schedule.direction.x *= -1

        if x_max is not None and self.center[0] > x_max:
            if self.schedule.on_boundary == "reset":
                self._reset_state()
            elif self.schedule.on_boundary == "oscillate":
                self.schedule.direction.x *= -1

    def _reset_state(self):
        shape = self.config.shape
        self.center = shape.center.to_numpy()

    def check_collision(self, center, radius) -> bool:
        # Calculate the squared distance between the centers of the two circles
        distance_sq = np.sum((self.center - center) ** 2)

        # Calculate the squared sum of the radii
        sum_radii_sq = (self.radius + radius) ** 2

        # If the distance is less than or equal to the sum of the radii, a collision occurs
        return bool(distance_sq <= sum_radii_sq)


class PolygonBoundary:
    def __init__(self, config: PolygonBoundaryConfig):
        self.vertices = [
            v.to_numpy() if isinstance(v, Vector2) else v for v in config.vertices
        ]
        self.walls = []
        for i in range(len(self.vertices)):
            p1 = self.vertices[i]
            p2 = self.vertices[(i + 1) % len(self.vertices)]
            self.walls.append((p1, p2))

    def violating_boundary(self, agent):
        for wall in self.walls:
            if self.is_colliding(agent, wall):
                return True
        return False

    def is_colliding(self, agent, wall):
        p1, p2 = wall
        agent_pos = agent.pos
        line_vec = p2 - p1
        point_vec = agent_pos - p1
        line_len_sq = np.sum(line_vec**2)

        t = np.dot(point_vec, line_vec) / line_len_sq
        t = np.clip(t, 0, 1)

        closest_point = p1 + t * line_vec
        dist_vec = agent_pos - closest_point
        dist_sq = np.sum(dist_vec**2)

        if dist_sq < agent.radius**2:
            return True

        return False
