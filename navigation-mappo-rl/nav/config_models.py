import numpy as np
import random
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator

PI = np.pi

"""
This file contains all the different config settings used to communicate between different modules of this repo
"""

# A basic vector class, also used to represent points
class Vector2(BaseModel):
    x: float
    y: float

    def to_numpy(self):
        return np.array([self.x, self.y])


class Line(BaseModel):
    start: Vector2
    end: Vector2


class Rectangle(BaseModel):
    type: Literal["rectangle"] = "rectangle"
    center: Vector2
    width: float
    height: float
    rotation: float = 0  # Rotation in degrees


class AgentConfig(BaseModel):
    start_pos: Rectangle # Exact point will be sampled
    goal_pos: Rectangle
    radius: float = 0.02
    max_speed: float
    agent_col: str = "blue"
    max_range: float = 0.25
    fov_degrees: float = 210.0


class Circle(BaseModel):
    type: Literal["circle"] = "circle"
    center: Vector2
    radius: float


class ObstacleSchedule(BaseModel):
    speed: Optional[float] = None
    direction: Optional[Vector2] = None
    angular_speed: Optional[float] = None # degrees
    rotating_up: Optional[bool] = None
    boundary_x_min: Optional[float] = None
    boundary_x_max: Optional[float] = None


class ObstacleConfig(BaseModel):
    shape: Union[Rectangle, Circle]
    schedule: Optional[ObstacleSchedule] = None
    noise: Optional[float] = None


class PolygonBoundaryConfig(BaseModel):
    type: Literal["polygon"] = "polygon"
    vertices: List[Vector2]


class EnvConfig(BaseModel):
    boundary: PolygonBoundaryConfig
    obstacles: List[ObstacleConfig] = []
    agents: List[AgentConfig]
    max_time: int
    num_rays: int = 60
    goal_threshold: float = 0.02
    repeat_steps: int = 2
    num_agents_per_group: int = 1
    state_image_size: int = 64
