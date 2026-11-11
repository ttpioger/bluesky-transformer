from .config_models import *
from .obstacles import RectangleObstacle, CircleObstacle, Obstacle
import numpy as np


class Line(BaseModel):
    p1: Vector2
    p2: Vector2


class Ray(BaseModel):
    origin: Vector2
    direction: Vector2
    length: float


IntersectingWith = Literal["obstacle", "boundary", "agent", "goal"]


class RayIntersectionOutput(BaseModel):
    intersects: bool
    intersection: Optional[Vector2] = None
    t: Optional[float] = None
    intersecting_with: Optional[IntersectingWith] = None


NoHit = RayIntersectionOutput(intersects=False)


# ===== VECTORIZED BATCH INTERSECTION FUNCTIONS =====


def batch_ray_circle_intersection(rays: np.ndarray, circles: np.ndarray) -> np.ndarray:
    """
    Vectorized ray-circle intersection for multiple rays and circles.

    Args:
        rays: [N, 5] array of [origin_x, origin_y, dir_x, dir_y, length]
        circles: [M, 3] array of [center_x, center_y, radius]

    Returns:
        distances: [N, M] array of distances (np.inf for no intersection)
    """
    if len(rays) == 0 or len(circles) == 0:
        return np.full((len(rays), len(circles)), np.inf)

    # Extract components with broadcasting shapes
    origins = rays[:, None, :2]  # [N, 1, 2]
    directions = rays[:, None, 2:4]  # [N, 1, 2]
    lengths = rays[:, None, 4]  # [N, 1]

    centers = circles[None, :, :2]  # [1, M, 2]
    radii = circles[None, :, 2]  # [1, M]

    # Vector from ray origin to circle center [N, M, 2]
    oc = origins - centers

    # Quadratic equation coefficients [N, M]
    a = np.sum(directions**2, axis=2)
    b = 2.0 * np.sum(oc * directions, axis=2)
    c = np.sum(oc**2, axis=2) - radii**2

    # Calculate discriminant [N, M]
    discriminant = b**2 - 4 * a * c

    # Only process valid intersections
    valid_mask = discriminant >= 0

    # Calculate intersection points (protect against negative discriminant)
    sqrt_disc = np.sqrt(np.maximum(discriminant, 0))
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    # Choose closest valid intersection [N, M]
    t1_valid = (t1 >= 0) & (t1 <= lengths) & valid_mask
    t2_valid = (t2 >= 0) & (t2 <= lengths) & valid_mask

    # Select the closest valid intersection
    t = np.where(t1_valid, t1, np.where(t2_valid, t2, np.inf))

    return t


def batch_ray_rectangle_intersection(
    rays: np.ndarray, rectangles: np.ndarray
) -> np.ndarray:
    """
    Vectorized ray-rectangle intersection for multiple rays and rectangles.

    Args:
        rays: [N, 5] array of [origin_x, origin_y, dir_x, dir_y, length]
        rectangles: [M, 5] array of [center_x, center_y, width, height, rotation_degrees]

    Returns:
        distances: [N, M] array of distances (np.inf for no intersection)
    """
    if len(rays) == 0 or len(rectangles) == 0:
        return np.full((len(rays), len(rectangles)), np.inf)

    N, M = len(rays), len(rectangles)
    distances = np.full((N, M), np.inf)

    # Extract ray components
    ray_origins = rays[:, :2]  # [N, 2]
    ray_directions = rays[:, 2:4]  # [N, 2]
    ray_lengths = rays[:, 4]  # [N]

    # Process each rectangle (vectorize across rays for each rectangle)
    for m in range(M):
        rect = rectangles[m]
        center = rect[:2]
        width, height = rect[2], rect[3]
        rotation = rect[4]

        # Transform all rays to rectangle's local coordinates
        local_origins = ray_origins - center[None, :]  # [N, 2]

        # Rotation matrix (negative to undo rectangle's rotation)
        theta = -np.radians(rotation)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

        # Rotate origins and directions
        local_origins = local_origins @ rotation_matrix.T  # [N, 2]
        local_directions = ray_directions @ rotation_matrix.T  # [N, 2]

        # Rectangle bounds
        half_width, half_height = width / 2, height / 2

        # Define rectangle edges as line segments
        edges = np.array(
            [
                [-half_width, -half_height, half_width, -half_height],  # bottom
                [half_width, -half_height, half_width, half_height],  # right
                [half_width, half_height, -half_width, half_height],  # top
                [-half_width, half_height, -half_width, -half_height],  # left
            ]
        )  # [4, 4] = [p1_x, p1_y, p2_x, p2_y]

        # Test all rays against all edges of this rectangle
        min_distances = np.full(N, np.inf)

        for edge in edges:
            edge_distances = batch_ray_line_intersection_raw(
                local_origins, local_directions, ray_lengths, edge[None, :]  # [1, 4]
            )  # [N, 1]

            valid_intersections = edge_distances[:, 0]
            min_distances = np.minimum(min_distances, valid_intersections)

        distances[:, m] = min_distances

    return distances


def batch_ray_line_intersection_raw(
    ray_origins: np.ndarray,
    ray_directions: np.ndarray,
    ray_lengths: np.ndarray,
    lines: np.ndarray,
) -> np.ndarray:
    """
    Raw vectorized ray-line intersection without object wrapping.

    Args:
        ray_origins: [N, 2] array
        ray_directions: [N, 2] array
        ray_lengths: [N] array
        lines: [M, 4] array of [p1_x, p1_y, p2_x, p2_y]

    Returns:
        distances: [N, M] array
    """
    if len(ray_origins) == 0 or len(lines) == 0:
        return np.full((len(ray_origins), len(lines)), np.inf)

    # Broadcast for vectorized computation
    origins = ray_origins[:, None, :]  # [N, 1, 2]
    directions = ray_directions[:, None, :]  # [N, 1, 2]
    lengths = ray_lengths[:, None]  # [N, 1]

    line_p1 = lines[None, :, :2]  # [1, M, 2]
    line_p2 = lines[None, :, 2:4]  # [1, M, 2]

    # Line segment vectors [1, M, 2]
    line_directions = line_p2 - line_p1

    # Vector from ray origin to line start [N, M, 2]
    origin_to_line = line_p1 - origins

    # Solve linear system using Cramer's rule
    # denominator = ray_dir.x * line_dir.y - ray_dir.y * line_dir.x
    denominator = (
        directions[:, :, 0] * line_directions[:, :, 1]
        - directions[:, :, 1] * line_directions[:, :, 0]
    )  # [N, M]

    # Check for parallel lines
    parallel_mask = np.abs(denominator) < 1e-10

    # Solve for t and s
    t = (
        origin_to_line[:, :, 0] * line_directions[:, :, 1]
        - origin_to_line[:, :, 1] * line_directions[:, :, 0]
    ) / np.where(
        parallel_mask, 1.0, denominator
    )  # [N, M]

    s = (
        origin_to_line[:, :, 0] * directions[:, :, 1]
        - origin_to_line[:, :, 1] * directions[:, :, 0]
    ) / np.where(
        parallel_mask, 1.0, denominator
    )  # [N, M]

    # Validity conditions
    valid = (
        ~parallel_mask
        & (t >= 0)
        & (t <= lengths)  # ray constraints
        & (s >= 0)
        & (s <= 1)
    )  # line segment constraints

    return np.where(valid, t, np.inf)


def batch_ray_line_intersection(rays: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """
    Vectorized ray-line intersection for multiple rays and line segments.

    Args:
        rays: [N, 5] array of [origin_x, origin_y, dir_x, dir_y, length]
        lines: [M, 4] array of [p1_x, p1_y, p2_x, p2_y]

    Returns:
        distances: [N, M] array of distances (np.inf for no intersection)
    """
    if len(rays) == 0 or len(lines) == 0:
        return np.full((len(rays), len(lines)), np.inf)

    return batch_ray_line_intersection_raw(
        rays[:, :2], rays[:, 2:4], rays[:, 4], lines  # origins  # directions  # lengths
    )


def batch_ray_intersection(
    rays: np.ndarray, obstacles: List[Obstacle], boundaries: List[PolygonBoundaryConfig]
) -> np.ndarray:
    """
    Main vectorized intersection function for multiple rays against all obstacles and boundaries.

    Args:
        rays: [N, 5] array of [origin_x, origin_y, dir_x, dir_y, length]
        obstacles: List of obstacle objects
        boundaries: List of boundary configurations

    Returns:
        distances: [N] array of closest intersection distances for each ray
    """
    if len(rays) == 0:
        return np.array([])

    N = len(rays)
    min_distances = np.full(N, np.inf)

    # Process circle obstacles
    circles = []
    rectangles = []

    for obstacle in obstacles:
        shape = obstacle.get_current_state()
        if isinstance(shape, Circle):
            circles.append([shape.center.x, shape.center.y, shape.radius])
        elif isinstance(shape, Rectangle):
            rectangles.append(
                [
                    shape.center.x,
                    shape.center.y,
                    shape.width,
                    shape.height,
                    shape.rotation,
                ]
            )

    # Batch process circles
    if circles:
        circle_array = np.array(circles)  # [M_circles, 3]
        circle_distances = batch_ray_circle_intersection(
            rays, circle_array
        )  # [N, M_circles]
        min_circle_distances = np.min(circle_distances, axis=1)  # [N]
        min_distances = np.minimum(min_distances, min_circle_distances)

    # Batch process rectangles
    if rectangles:
        rectangle_array = np.array(rectangles)  # [M_rects, 5]
        rect_distances = batch_ray_rectangle_intersection(
            rays, rectangle_array
        )  # [N, M_rects]
        min_rect_distances = np.min(rect_distances, axis=1)  # [N]
        min_distances = np.minimum(min_distances, min_rect_distances)

    # Process boundaries
    for boundary in boundaries:
        from .obstacles import PolygonBoundary

        polygon = PolygonBoundary(boundary)

        # Convert walls to line array
        walls = []
        for wall in polygon.walls:
            p1, p2 = wall
            walls.append([p1[0], p1[1], p2[0], p2[1]])

        if walls:
            wall_array = np.array(walls)  # [M_walls, 4]
            wall_distances = batch_ray_line_intersection(
                rays, wall_array
            )  # [N, M_walls]
            min_wall_distances = np.min(wall_distances, axis=1)  # [N]
            min_distances = np.minimum(min_distances, min_wall_distances)

    return min_distances


def batch_ray_agent_intersection(
    rays: np.ndarray,
    agents: List[Circle],
    rays_per_agent: List[int],
) -> np.ndarray:
    """
    Vectorized ray-agent intersection that excludes self-intersection.

    Args:
        rays: [N, 5] array of [origin_x, origin_y, dir_x, dir_y, length]
        agents: List of agent circles
        rays_per_agent: List of ray counts per agent to map rays to agents

    Returns:
        distances: [N] array of closest intersection distances for each ray
    """
    if len(rays) == 0 or len(agents) == 0:
        return np.full(len(rays), np.inf)

    N = len(rays)
    min_distances = np.full(N, np.inf)

    ray_start_idx = 0
    for agent_idx, num_rays in enumerate(rays_per_agent):
        # Get rays for this specific agent
        agent_rays = rays[ray_start_idx : ray_start_idx + num_rays]

        # Create list of all other agents (excluding current agent)
        other_agents = [agent for i, agent in enumerate(agents) if i != agent_idx]

        if other_agents:
            # Convert other agents to numpy array format for batch processing
            other_agents_array = np.array(
                [
                    [agent.center.x, agent.center.y, agent.radius]
                    for agent in other_agents
                ]
            )  # [M_other_agents, 3]

            # Check intersections for this agent's rays against all other agents
            agent_distances = batch_ray_circle_intersection(
                agent_rays, other_agents_array
            )  # [num_rays, M_other_agents]

            # Find minimum distance for each ray among all other agents
            min_agent_distances = np.min(agent_distances, axis=1)  # [num_rays]

            # Update global distances array
            global_indices = slice(ray_start_idx, ray_start_idx + num_rays)
            min_distances[global_indices] = np.minimum(
                min_distances[global_indices], min_agent_distances
            )

        ray_start_idx += num_rays

    return min_distances


def batch_ray_intersection_detailed(
    rays: np.ndarray,
    obstacles: List[Obstacle],
    boundaries: List[PolygonBoundaryConfig],
    goals: Optional[List[Optional[Circle]]] = None,
    agents: Optional[List[Circle]] = None,
    rays_per_agent: Optional[List[int]] = None,
) -> List[RayIntersectionOutput]:
    """
    Detailed vectorized intersection function that returns full RayIntersectionOutput for each ray.

    Args:
        rays: [N, 5] array of [origin_x, origin_y, dir_x, dir_y, length]
        obstacles: List of obstacle objects
        boundaries: List of boundary configurations
        goals: List of goal circles, one per agent (can contain None for agents without goals)
        rays_per_agent: List of ray counts per agent to map rays to correct goal rectangles

    Returns:
        results: List[RayIntersectionOutput] - one result per ray with full details
    """
    if len(rays) == 0:
        return []

    N = len(rays)
    results = []

    # Initialize tracking arrays for each ray
    closest_distances = np.full(N, np.inf)
    closest_intersections = [None] * N
    intersecting_with = [None] * N

    # Separate obstacles by type
    circle_obstacles = []
    rectangle_obstacles = []

    for obstacle in obstacles:
        shape = obstacle.get_current_state()
        if isinstance(shape, Circle):
            circle_obstacles.append((obstacle, shape))
        elif isinstance(shape, Rectangle):
            rectangle_obstacles.append((obstacle, shape))

    # Process circle obstacles
    if circle_obstacles:
        circles = np.array(
            [
                [shape.center.x, shape.center.y, shape.radius]
                for _, shape in circle_obstacles
            ]
        )

        circle_distances = batch_ray_circle_intersection(
            rays, circles
        )  # [N, M_circles]

        # Find closest intersection for each ray among all circles
        for i in range(N):
            min_idx = np.argmin(circle_distances[i])
            min_dist = circle_distances[i, min_idx]

            if min_dist < closest_distances[i]:
                closest_distances[i] = min_dist
                intersecting_with[i] = "obstacle"

                # Calculate intersection point
                ray_origin = rays[i, :2]
                ray_direction = rays[i, 2:4]
                intersection_point = ray_origin + ray_direction * min_dist
                closest_intersections[i] = Vector2(
                    x=intersection_point[0], y=intersection_point[1]
                )

    # Process rectangle obstacles
    if rectangle_obstacles:
        rectangles = np.array(
            [
                [
                    shape.center.x,
                    shape.center.y,
                    shape.width,
                    shape.height,
                    shape.rotation,
                ]
                for _, shape in rectangle_obstacles
            ]
        )

        rect_distances = batch_ray_rectangle_intersection(
            rays, rectangles
        )  # [N, M_rects]

        # Find closest intersection for each ray among all rectangles
        for i in range(N):
            min_idx = np.argmin(rect_distances[i])
            min_dist = rect_distances[i, min_idx]

            if min_dist < closest_distances[i]:
                closest_distances[i] = min_dist
                intersecting_with[i] = "obstacle"

                # Calculate intersection point
                ray_origin = rays[i, :2]
                ray_direction = rays[i, 2:4]
                intersection_point = ray_origin + ray_direction * min_dist
                closest_intersections[i] = Vector2(
                    x=intersection_point[0], y=intersection_point[1]
                )

    # Process boundaries
    for boundary in boundaries:
        from .obstacles import PolygonBoundary

        polygon = PolygonBoundary(boundary)

        # Convert walls to line array
        walls = []
        for wall in polygon.walls:
            p1, p2 = wall
            walls.append([p1[0], p1[1], p2[0], p2[1]])

        if walls:
            wall_array = np.array(walls)
            wall_distances = batch_ray_line_intersection(
                rays, wall_array
            )  # [N, M_walls]

            # Find closest intersection for each ray among all walls
            for i in range(N):
                min_idx = np.argmin(wall_distances[i])
                min_dist = wall_distances[i, min_idx]

                if min_dist < closest_distances[i]:
                    closest_distances[i] = min_dist
                    intersecting_with[i] = "boundary"

                    # Calculate intersection point
                    ray_origin = rays[i, :2]
                    ray_direction = rays[i, 2:4]
                    intersection_point = ray_origin + ray_direction * min_dist
                    closest_intersections[i] = Vector2(
                        x=intersection_point[0], y=intersection_point[1]
                    )

    # Process goal rectangles (agent-specific)
    if goals is not None and rays_per_agent is not None:
        ray_start_idx = 0
        for agent_idx, (goal, num_rays) in enumerate(zip(goals, rays_per_agent)):
            if goal is None:
                ray_start_idx += num_rays
                continue

            # Get rays for this specific agent
            agent_rays = rays[ray_start_idx : ray_start_idx + num_rays]

            # Convert goal rectangle to numpy array format for batch processing
            goal_circle_array = np.array(
                [
                    [
                        goal.center.x,
                        goal.center.y,
                        goal.radius,
                    ]
                ]
            )  # [1, 5]

            # Check intersections for this agent's rays only
            goal_circle_distances = batch_ray_circle_intersection(
                agent_rays, goal_circle_array
            )  # [num_rays, 1]

            # Update closest intersections for this agent's rays
            for i in range(num_rays):
                global_ray_idx = ray_start_idx + i
                min_dist = goal_circle_distances[
                    i, 0
                ]  # Get distance from the single goal circle

                if min_dist < closest_distances[global_ray_idx]:
                    closest_distances[global_ray_idx] = min_dist
                    intersecting_with[global_ray_idx] = "goal"

                    # Calculate intersection point
                    ray_origin = rays[global_ray_idx, :2]
                    ray_direction = rays[global_ray_idx, 2:4]
                    intersection_point = ray_origin + ray_direction * min_dist
                    closest_intersections[global_ray_idx] = Vector2(
                        x=intersection_point[0], y=intersection_point[1]
                    )

            ray_start_idx += num_rays

    # Process agents (exclude self-intersection)
    if agents is not None and rays_per_agent is not None:
        ray_start_idx = 0
        for agent_idx, num_rays in enumerate(rays_per_agent):
            # Get rays for this specific agent
            agent_rays = rays[ray_start_idx : ray_start_idx + num_rays]

            # Create list of all other agents (excluding current agent)
            other_agents = [agent for i, agent in enumerate(agents) if i != agent_idx]

            if other_agents:
                # Convert other agents to numpy array format for batch processing
                other_agents_array = np.array(
                    [
                        [agent.center.x, agent.center.y, agent.radius]
                        for agent in other_agents
                    ]
                )  # [M_other_agents, 3]

                # Check intersections for this agent's rays against all other agents
                agent_distances = batch_ray_circle_intersection(
                    agent_rays, other_agents_array
                )  # [num_rays, M_other_agents]

                # Update closest intersections for this agent's rays
                for i in range(num_rays):
                    global_ray_idx = ray_start_idx + i
                    min_idx = np.argmin(agent_distances[i])
                    min_dist = agent_distances[i, min_idx]

                    if min_dist < closest_distances[global_ray_idx]:
                        closest_distances[global_ray_idx] = min_dist
                        intersecting_with[global_ray_idx] = "agent"

                        # Calculate intersection point
                        ray_origin = rays[global_ray_idx, :2]
                        ray_direction = rays[global_ray_idx, 2:4]
                        intersection_point = ray_origin + ray_direction * min_dist
                        closest_intersections[global_ray_idx] = Vector2(
                            x=intersection_point[0], y=intersection_point[1]
                        )

            ray_start_idx += num_rays

    # Build RayIntersectionOutput objects for each ray
    for i in range(N):
        if closest_distances[i] < np.inf:
            results.append(
                RayIntersectionOutput(
                    intersects=True,
                    intersection=closest_intersections[i],
                    t=closest_distances[i],
                    intersecting_with=intersecting_with[i],
                )
            )
        else:
            results.append(NoHit)

    return results


# ===== HELPER FUNCTIONS FOR EASY CONVERSION =====


def rays_to_array(ray_list: List[Ray]) -> np.ndarray:
    """Convert list of Ray objects to numpy array format."""
    if not ray_list:
        return np.empty((0, 5))

    rays_array = np.zeros((len(ray_list), 5))
    for i, ray in enumerate(ray_list):
        rays_array[i] = [
            ray.origin.x,
            ray.origin.y,
            ray.direction.x,
            ray.direction.y,
            ray.length,
        ]
    return rays_array


def create_lidar_rays(
    origin: Union[Vector2, np.ndarray],
    base_direction: Union[Vector2, np.ndarray],
    num_rays: int = 180,
    max_range: float = 10.0,
    fov_degrees: float = 360.0,
) -> np.ndarray:
    """
    Create a batch of LiDAR rays for vectorized processing.

    Args:
        origin: Starting point for all rays
        base_direction: Base direction vector (will be normalized)
        num_rays: Number of rays to generate
        max_range: Maximum range for all rays
        fov_degrees: Field of view in degrees

    Returns:
        rays: [num_rays, 5] array ready for batch processing
    """
    # Normalize base direction
    if isinstance(base_direction, Vector2):
        base_dir = np.array([base_direction.x, base_direction.y])
    else:
        base_dir = base_direction

    base_dir = base_dir / np.linalg.norm(base_dir)

    # Calculate base angle
    base_angle = np.arctan2(base_dir[1], base_dir[0])

    # Generate angles
    start_angle = base_angle - np.radians(fov_degrees / 2)
    end_angle = base_angle + np.radians(fov_degrees / 2)

    if num_rays == 1:
        angles = np.array([base_angle])
    else:
        angles = np.linspace(start_angle, end_angle, num_rays)

    # Create rays array
    rays = np.zeros((num_rays, 5))

    origin = origin.to_numpy() if isinstance(origin, Vector2) else origin
    rays[:, 0] = origin[0]  # origin_x
    rays[:, 1] = origin[1]  # origin_y
    rays[:, 2] = np.cos(angles)  # direction_x
    rays[:, 3] = np.sin(angles)  # direction_y
    rays[:, 4] = max_range  # length

    return rays


# ===== ORIGINAL SINGLE-RAY FUNCTIONS (PRESERVED FOR COMPATIBILITY) =====


def ray_circle_intersection(ray: Ray, circle: Circle) -> RayIntersectionOutput:
    """
    Compute intersection between a ray and a circle using quadratic formula.
    Returns the closest intersection point along the ray (t >= 0 and t <= ray.length).
    """
    # Convert to numpy arrays for easier computation
    origin = ray.origin.to_numpy()
    direction = ray.direction.to_numpy()
    center = circle.center.to_numpy()
    radius = circle.radius

    # Vector from ray origin to circle center
    oc = origin - center

    # Quadratic equation coefficients: at² + bt + c = 0
    a = np.dot(direction, direction)
    b = 2.0 * np.dot(oc, direction)
    c = np.dot(oc, oc) - radius * radius

    # Calculate discriminant
    discriminant = b * b - 4 * a * c

    # No intersection if discriminant is negative
    if discriminant < 0:
        return NoHit

    # Calculate the two possible intersection points
    sqrt_discriminant = np.sqrt(discriminant)
    t1 = (-b - sqrt_discriminant) / (2.0 * a)
    t2 = (-b + sqrt_discriminant) / (2.0 * a)

    # We want the closest intersection point that's in front of the ray (t >= 0) and within ray length (t <= ray.length)
    if t1 >= 0 and t1 <= ray.length:
        t = t1
    elif t2 >= 0 and t2 <= ray.length:
        t = t2
    else:
        # Both intersections are either behind the ray origin or beyond ray length
        return NoHit

    # Calculate intersection point
    intersection_point = origin + t * direction

    return RayIntersectionOutput(
        intersects=True,
        intersection=Vector2(x=intersection_point[0], y=intersection_point[1]),
        t=t,
    )


def ray_rectangle_intersection(ray: Ray, rectangle: Rectangle) -> RayIntersectionOutput:
    """
    Compute intersection between a ray and a rotated rectangle.
    Transform ray to rectangle's local coordinate system and check against edges.
    """
    # Convert to numpy arrays
    origin = ray.origin.to_numpy()
    direction = ray.direction.to_numpy()
    center = rectangle.center.to_numpy()

    # Transform ray to rectangle's local coordinate system
    # Translate to rectangle center
    local_origin = origin - center

    # Rotate by negative rotation to undo rectangle's rotation
    theta = -np.radians(rectangle.rotation)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    local_origin = rotation_matrix @ local_origin
    local_direction = rotation_matrix @ direction

    # Rectangle bounds in local coordinate system
    half_width = rectangle.width / 2
    half_height = rectangle.height / 2

    # Check intersection with all four edges of the rectangle
    closest_t = float("inf")
    closest_intersection = None

    # Define the four edges of the rectangle
    edges = [
        # Bottom edge
        Line(
            p1=Vector2(x=-half_width, y=-half_height),
            p2=Vector2(x=half_width, y=-half_height),
        ),
        # Right edge
        Line(
            p1=Vector2(x=half_width, y=-half_height),
            p2=Vector2(x=half_width, y=half_height),
        ),
        # Top edge
        Line(
            p1=Vector2(x=half_width, y=half_height),
            p2=Vector2(x=-half_width, y=half_height),
        ),
        # Left edge
        Line(
            p1=Vector2(x=-half_width, y=half_height),
            p2=Vector2(x=-half_width, y=-half_height),
        ),
    ]

    # Create ray in local coordinates (preserve the length)
    local_ray = Ray(
        origin=Vector2(x=local_origin[0], y=local_origin[1]),
        direction=Vector2(x=local_direction[0], y=local_direction[1]),
        length=ray.length,
    )

    # Check intersection with each edge
    for edge in edges:
        result = ray_line_intersection(local_ray, edge)
        if (
            result.intersects
            and result.t is not None
            and result.t < closest_t
            and result.t >= 0
            and result.t <= ray.length
        ):
            closest_t = result.t
            closest_intersection = result.intersection

    if closest_intersection is None:
        return NoHit

    # Transform intersection point back to world coordinates
    local_intersection = np.array([closest_intersection.x, closest_intersection.y])

    # Rotate back to world coordinates
    inverse_rotation_matrix = np.array(
        [[cos_theta, sin_theta], [-sin_theta, cos_theta]]
    )

    world_intersection = inverse_rotation_matrix @ local_intersection + center

    return RayIntersectionOutput(
        intersects=True,
        intersection=Vector2(x=world_intersection[0], y=world_intersection[1]),
        t=closest_t,
    )


def ray_line_intersection(ray: Ray, line: Line) -> RayIntersectionOutput:
    """
    Compute intersection between a ray and a line segment.
    Uses parametric line equations and solves for intersection.
    """
    # Convert to numpy arrays
    ray_origin = ray.origin.to_numpy()
    ray_direction = ray.direction.to_numpy()
    line_p1 = line.p1.to_numpy()
    line_p2 = line.p2.to_numpy()

    # Line segment vector
    line_direction = line_p2 - line_p1

    # Vector from ray origin to line start
    origin_to_line = line_p1 - ray_origin

    # Solve the system:
    # ray_origin + t * ray_direction = line_p1 + s * line_direction
    # This gives us: t * ray_direction - s * line_direction = origin_to_line

    # Create coefficient matrix
    # [ray_direction.x, -line_direction.x] [t] = [origin_to_line.x]
    # [ray_direction.y, -line_direction.y] [s]   [origin_to_line.y]

    denominator = ray_direction[0] * (-line_direction[1]) - ray_direction[1] * (
        -line_direction[0]
    )
    denominator = (
        ray_direction[0] * line_direction[1] - ray_direction[1] * line_direction[0]
    )

    # Lines are parallel
    if abs(denominator) < 1e-10:
        return NoHit

    # Solve for t and s using Cramer's rule
    t = (
        origin_to_line[0] * line_direction[1] - origin_to_line[1] * line_direction[0]
    ) / denominator
    s = (
        origin_to_line[0] * ray_direction[1] - origin_to_line[1] * ray_direction[0]
    ) / denominator

    # Check if intersection is valid:
    # t >= 0 (intersection is in front of ray)
    # t <= ray.length (intersection is within ray length)
    # 0 <= s <= 1 (intersection is within line segment)
    if t >= 0 and t <= ray.length and 0 <= s <= 1:
        # Calculate intersection point
        intersection_point = ray_origin + t * ray_direction

        return RayIntersectionOutput(
            intersects=True,
            intersection=Vector2(x=intersection_point[0], y=intersection_point[1]),
            t=t,
        )

    return NoHit


def ray_obstacle_intersection(ray: Ray, obstacle: Obstacle) -> RayIntersectionOutput:
    obstacle_type = obstacle.config.shape
    if isinstance(obstacle_type, Rectangle):
        return ray_rectangle_intersection(ray, obstacle_type)
    elif isinstance(obstacle_type, Circle):
        return ray_circle_intersection(ray, obstacle_type)
    else:
        raise ValueError(f"Unknown obstacle type: {obstacle_type}")


def ray_boundary_intersection(
    ray: Ray, boundary: PolygonBoundaryConfig
) -> RayIntersectionOutput:
    """
    Find the closest intersection between a ray and polygon boundary walls.
    """
    # Create a PolygonBoundary object to get the walls
    from .obstacles import PolygonBoundary

    polygon = PolygonBoundary(boundary)

    closest_t = float("inf")
    closest_intersection = None

    for wall in polygon.walls:
        p1, p2 = wall
        # Convert numpy arrays back to Vector2 for consistency
        line = Line(p1=Vector2(x=p1[0], y=p1[1]), p2=Vector2(x=p2[0], y=p2[1]))

        result = ray_line_intersection(ray, line)
        if (
            result.intersects
            and result.t is not None
            and result.t < closest_t
            and result.t <= ray.length
        ):
            closest_t = result.t
            closest_intersection = result.intersection

    if closest_intersection is None:
        return NoHit

    return RayIntersectionOutput(
        intersects=True, intersection=closest_intersection, t=closest_t
    )


def ray_intersection(
    ray: Ray,
    obstacles: List[Obstacle],
    boundaries: List[PolygonBoundaryConfig],
    goal_rectangle: Optional[Rectangle] = None,
) -> RayIntersectionOutput:
    """
    Find the closest intersection between a ray and obstacles and boundaries.
    """

    closest_t = float("inf")
    closest_intersection = None
    intersecting_with = None

    for obstacle in obstacles:
        result = ray_obstacle_intersection(ray, obstacle)
        if result.intersects and result.t is not None and result.t < closest_t:
            closest_t = result.t
            closest_intersection = result.intersection
            intersecting_with = "obstacle"

    for boundary in boundaries:
        result = ray_boundary_intersection(ray, boundary)
        if result.intersects and result.t is not None and result.t < closest_t:
            closest_t = result.t
            closest_intersection = result.intersection
            intersecting_with = "boundary"

    if goal_rectangle is not None:
        result = ray_rectangle_intersection(ray, goal_rectangle)
        if result.intersects and result.t is not None and result.t < closest_t:
            closest_t = result.t
            closest_intersection = result.intersection
            intersecting_with = "goal"

    if closest_intersection is None:
        return NoHit

    return RayIntersectionOutput(
        intersects=True,
        intersection=closest_intersection,
        t=closest_t,
        intersecting_with=intersecting_with,
    )
