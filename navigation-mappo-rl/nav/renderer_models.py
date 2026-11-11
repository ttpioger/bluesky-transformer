from pydantic import BaseModel
from typing import List, Tuple, Dict, Union, Optional
from .config_models import *
from .ray_intersection import RayIntersectionOutput


class AgentState(BaseModel):
    position: Tuple[float, float]
    radius: float
    color: str
    velocity: Tuple[float, float]
    direction: Tuple[float, float]  # Unit vector representing facing direction
    lidar_observation: list[RayIntersectionOutput]
    fov_degrees: float
    max_range: float
    goals: Circle
    goal_reached: bool
    last_reward: float


class ObstacleState(BaseModel):
    shape: str  # "rectangle", "circle"
    properties: Dict


class BoundaryState(BaseModel):
    vertices: List[Tuple[float, float]]


class RenderState(BaseModel):
    """
    A complete, serializable snapshot of the environment for a single frame.
    This object is renderer-agnostic.
    """

    agents: List[AgentState]
    obstacles: List[Union[Rectangle, Circle]]
    boundary: BoundaryState
