from typing import Tuple
import numpy as np
from .config_models import Rectangle


def convert_to_polar(a: np.ndarray) -> Tuple[float, np.ndarray]:
    magnitude = float(np.linalg.norm(a))
    if magnitude > 1e-10:  # Avoid division by zero
        unit_vector = a / magnitude
    else:
        # Return zero vector or default direction when magnitude is zero
        unit_vector = np.array([1.0, 0.0])  # Default to pointing right
        magnitude = 0.0
    return magnitude, unit_vector


def circle_line_intersection(
    circle_center: np.ndarray,
    circle_radius: float,
    line_start: np.ndarray,
    line_end: np.ndarray,
) -> bool:
    """
    Check if a circle intersects with a line segment.

    Args:
        circle_center: Center point of the circle [x, y]
        circle_radius: Radius of the circle
        line_start: Start point of the line segment [x, y]
        line_end: End point of the line segment [x, y]

    Returns:
        bool: True if circle intersects with line segment, False otherwise
    """
    # Vector from line start to line end
    line_vec = line_end - line_start

    # Vector from line start to circle center
    start_to_center = circle_center - line_start

    # Calculate the length of the line segment
    line_length_sq = np.dot(line_vec, line_vec)

    # Handle degenerate case where line is actually a point
    if line_length_sq < 1e-10:
        # Line is a point, check distance from circle center to that point
        distance_to_point = np.linalg.norm(start_to_center)
        return distance_to_point <= circle_radius

    # Project circle center onto the line (parameterized by t)
    # t = 0 is line_start, t = 1 is line_end
    t = np.dot(start_to_center, line_vec) / line_length_sq

    # Clamp t to [0, 1] to stay within the line segment
    t = max(0.0, min(1.0, t))

    # Find the closest point on the line segment to the circle center
    closest_point = line_start + t * line_vec

    # Calculate distance from circle center to closest point
    distance_to_closest = np.linalg.norm(circle_center - closest_point)

    # Check if circle intersects (distance <= radius)
    return distance_to_closest <= circle_radius


def circle_rectangle_intersection(
    circle_center: np.ndarray,
    circle_radius: float,
    rect_center: np.ndarray,
    rect_width: float,
    rect_height: float,
    rect_rotation: float = 0.0,
) -> bool:
    """
    Check if a circle intersects with a rotated rectangle.

    Args:
        circle_center: Center point of the circle [x, y]
        circle_radius: Radius of the circle
        rect_center: Center point of the rectangle [x, y]
        rect_width: Width of the rectangle
        rect_height: Height of the rectangle
        rect_rotation: Rotation of the rectangle in degrees

    Returns:
        bool: True if circle intersects with rectangle, False otherwise
    """
    # Transform circle center to rectangle's local coordinate system
    # Translate to rectangle center
    local_center = circle_center - rect_center

    # Rotate by negative rotation to undo rectangle's rotation
    theta = -np.radians(rect_rotation)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Apply rotation matrix
    rotated_center = np.array(
        [
            local_center[0] * cos_theta - local_center[1] * sin_theta,
            local_center[0] * sin_theta + local_center[1] * cos_theta,
        ]
    )

    # Rectangle bounds in local coordinate system
    half_width = rect_width / 2
    half_height = rect_height / 2

    # Find closest point on rectangle to circle center
    closest_x = max(-half_width, min(half_width, rotated_center[0]))
    closest_y = max(-half_height, min(half_height, rotated_center[1]))
    closest_point = np.array([closest_x, closest_y])

    # Calculate distance from circle center to closest point on rectangle
    distance = np.linalg.norm(rotated_center - closest_point)

    # Check if circle intersects (distance <= radius)
    return distance <= circle_radius


def sample_point_in_rectangle(rectangle: Rectangle) -> np.ndarray:
    return np.array(
        [
            np.random.uniform(
                rectangle.center.x - rectangle.width / 2,
                rectangle.center.x + rectangle.width / 2,
            ),
            np.random.uniform(
                rectangle.center.y - rectangle.height / 2,
                rectangle.center.y + rectangle.height / 2,
            ),
        ]
    )
