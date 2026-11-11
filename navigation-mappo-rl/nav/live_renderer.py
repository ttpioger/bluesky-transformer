import json
import arcade
import time
from arcade.color import BLACK, BLACK_BEAN, DARK_GRAY, GRAY
import numpy as np
from typing import List, Optional
from .renderer_models import RenderState, AgentState
import pickle

try:
    import imageio

    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

# --- Theme Colors ---
BOUNDARY_COLOR = arcade.color.WHITE
ARENA_COLOR = arcade.color.DIM_GRAY
BACKGROUND_COLOR = arcade.color.BLACK
OBSTACLE_COLOR = arcade.color.LIGHT_RED_OCHRE
OBSTACLE_BORDER_COLOR = arcade.color.BLACK
AGENT_COLOR = arcade.color.AERO_BLUE
AGENT_BORDER_COLOR = arcade.color.WHITE
AGENT_ARROW_COLOR = arcade.color.DARK_BLUE
AGENT_GOAL_COLOR = arcade.color.GREEN
POSITION_HISTORY_COLOR = (255, 255, 255, 100)

# --- Ray Colors ---
RAY_OBSTACLE_HIT_COLOR = arcade.color.CRIMSON
RAY_BOUNDARY_HIT_COLOR = arcade.color.DARK_ORANGE
RAY_GOAL_HIT_COLOR = arcade.color.LIME_GREEN
RAY_MISS_COLOR = arcade.color.LIGHT_STEEL_BLUE
RAY_INTERSECTION_MARKER_COLOR = arcade.color.WHITE
RAY_AGENT_HIT_COLOR = arcade.color.BLUE

DRAW_RAYS = False
DRAW_POSITION_HISTORY = False
RAY_ALPHA = 80
RAY_LINE_WIDTH = 2
DRAW_INTERSECTION_MARKERS = True
ANIMATE_RAYS = True
DRAW_AGENT_SHADOW = False

class SimulationWindow(arcade.Window):
    """
    Renders a simulation state using Arcade.
    """

    def __init__(
        self,
        window_width: int = 800,
        window_height: int = 800,
        target_fps: int = 30,
        record: bool = False,
        headless: bool = False,
    ):
        super().__init__(
            window_width, window_height, "Navigation Simulation", visible=not headless
        )
        arcade.set_background_color(BACKGROUND_COLOR)
        self.current_state = None
        self.cursor_pos = (0.0, 0.0)
        self.headless = headless

        # Set target FPS
        self.target_fps = target_fps
        # self.set_vsync(False)  # Disable VSync to allow custom FPS control
        self.set_update_rate(1 / target_fps)

        # Recording setup
        if record and not IMAGEIO_AVAILABLE:
            print("❌ Error: Recording requested but imageio not available.")
            print("   Install with: pip install imageio[ffmpeg]")
            print("   Running without recording...")

        self.record = record and IMAGEIO_AVAILABLE
        self.data = []
        self.frames = []
        self.recording_writer: Optional[imageio.core.Format.Writer] = None
        if self.record:
            print(
                f"🎬 Recording setup: {window_width}x{window_height} at {target_fps} FPS"
            )

        # FPS tracking
        self.frame_count = 0
        self.keys_pressed = set()
        self.position_history = {}
        self.paused = False

    def on_mouse_motion(self, x, y, dx, dy):
        self.cursor_pos = (x / self.width, y / self.height)

    def on_key_press(self, key, modifiers):
        """Handle keyboard input."""
        self.keys_pressed.add(key)
        if key == arcade.key.ESCAPE:
            # Quick exit
            self.close()
        if key == arcade.key.L:
            global DRAW_RAYS
            DRAW_RAYS = not DRAW_RAYS
        if key == arcade.key.P:
            self.paused = not self.paused

    def on_key_release(self, key, modifiers):
        """Handle keyboard key release."""
        if key in self.keys_pressed:
            self.keys_pressed.remove(key)

    def on_mouse_press(self, x, y, b, d):
        """Called whenever the mouse moves."""
        self.mouse_x = x
        self.mouse_y = y
        print(x, y)

    def get_mouse_position(self):
        return self.cursor_pos

    def get_human_action(self) -> List[float]:
        """
        Get the current action based on pressed keys.
        Returns: [vx, vy] where values are -1, 0, or 1
        """
        vx = 0.0
        vy = 0.0

        if arcade.key.UP in self.keys_pressed:
            vy += 1.0
        if arcade.key.DOWN in self.keys_pressed:
            vy -= 1.0
        if arcade.key.LEFT in self.keys_pressed:
            vx -= 1.0
        if arcade.key.RIGHT in self.keys_pressed:
            vx += 1.0

        return [vx, vy]

    def render(self, state: RenderState):
        """
        Receives a new state and schedules a redraw.
        """
        if self.paused:
            return
        self.current_state = state
        for idx, agent in enumerate(self.current_state.agents):
            if idx not in self.position_history:
                self.position_history[idx] = []
            self.position_history[idx].append(agent.position)

        if self.headless:
            # For headless rendering, just draw directly without events/flip
            self.on_draw()
        else:
            # For interactive rendering, handle events and flip buffer
            self.dispatch_events()
            self.on_draw()
            self.flip()

    def get_rgb_array(self) -> np.ndarray:
        """
        Get the current frame as an RGB array.
        """
        try:
            # Get the current frame as RGB data
            image = arcade.get_image()
            # Convert PIL image to numpy array
            frame = np.array(image)
            # Convert RGBA to RGB if necessary
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]
            return frame
        except Exception as e:
            print(f"Warning: Failed to capture RGB array: {e}")
            # Return a black frame as fallback
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def on_draw(self):
        """
        The main rendering loop.
        """
        self.clear()

        # Safety check - don't draw if no state is available
        if not self.current_state:
            return

        # --- Draw Boundary ---
        scaled_vertices = [
            (v[0] * self.width, v[1] * self.height)
            for v in self.current_state.boundary.vertices
        ]
        arcade.draw_polygon_outline(scaled_vertices, BOUNDARY_COLOR, line_width=10)
        arcade.draw_polygon_filled(scaled_vertices, ARENA_COLOR)

        # --- Draw Obstacles ---
        for obstacle in self.current_state.obstacles:
            if obstacle.type == "rectangle":
                x = obstacle.center.x * self.width
                y = obstacle.center.y * self.height
                width = obstacle.width * self.width
                height = obstacle.height * self.height
                rotation = (
                    -obstacle.rotation
                )  # Rotation in degrees (arcade expects degrees)
                rect = arcade.XYWH(x, y, width, height)

                arcade.draw_rect_filled(rect, OBSTACLE_COLOR, rotation)
                arcade.draw_rect_outline(rect, OBSTACLE_BORDER_COLOR, 4, rotation)

            elif obstacle.type == "circle":
                x = obstacle.center.x * self.width
                y = obstacle.center.y * self.height
                radius = obstacle.radius * self.width

                arcade.draw_circle_filled(x, y, radius, OBSTACLE_COLOR)
                arcade.draw_circle_outline(x, y, radius, OBSTACLE_BORDER_COLOR, 4)

        if DRAW_RAYS:
            # --- Draw Rays (behind agents) ---
            for agent in self.current_state.agents:
                self._draw_agent_rays(agent)

        if DRAW_POSITION_HISTORY:
            position_history_points = []
            for agent_id, position_history in self.position_history.items():
                for i in range(len(position_history) - 1):
                    x = position_history[i][0] * self.width
                    y = position_history[i][1] * self.height
                    position_history_points.append((x, y))

            arcade.draw_points(position_history_points, POSITION_HISTORY_COLOR, size=2)

            # for _, position_history in self.position_history.items():
            #     arcade.draw_circle_outline(
            #         position_history[-1][0] * self.width,
            #         position_history[-1][1] * self.height,
            #         agent.radius * self.width / 2,
            #         AGENT_COLOR,
            #         1,
            #     )

        # --- Draw Agents (on top of rays) ---
        for agent in self.current_state.agents:
            x = agent.position[0] * self.width
            y = agent.position[1] * self.height
            vX = agent.velocity[0] / 5 * self.width
            vY = agent.velocity[1] / 5 * self.height
            radius = agent.radius * self.width

            # Get agent colors based on config
            if agent.color and agent.color.lower() != "none":
                try:
                    agent_color = getattr(arcade.color, agent.color.upper())
                    # Create a lighter shade for goal by mixing with white
                    goal_color = tuple(
                        min(255, int(c * 0.7 + 255 * 0.3)) for c in agent_color[:3]
                    )
                except AttributeError:
                    # Fallback to default colors if the color name is not found
                    agent_color = AGENT_COLOR
                    goal_color = AGENT_GOAL_COLOR[:3]
            else:
                # Use default colors when no color is specified
                agent_color = AGENT_COLOR
                goal_color = AGENT_GOAL_COLOR[:3]

            agent_border_color = AGENT_BORDER_COLOR

            # Draw goal circle first (behind agent)
            goal_x = agent.goals.center.x * self.width
            goal_y = agent.goals.center.y * self.height
            goal_radius = agent.goals.radius * self.width

            # Ensure proper color formatting
            goal_fill_color = (*goal_color, 100)  # RGB + alpha
            goal_outline_color = goal_color  # RGB only for outline

            arcade.draw_circle_filled(
                goal_x, goal_y, goal_radius, goal_fill_color
            )  # Semi-transparent fill
            arcade.draw_circle_outline(
                goal_x, goal_y, goal_radius, goal_outline_color, 3
            )  # Solid outline

            if DRAW_AGENT_SHADOW:
                factor = 0.1
                arcade.draw_circle_filled(
                x - (factor)*vX, 
                    y - (factor)*vY, 
                    radius + factor*(vY), BLACK_BEAN
                )
            # Draw agent circle
            arcade.draw_circle_filled(x, y, radius, agent_color)
            arcade.draw_circle_outline(x, y, radius, agent_border_color, 1)

            # Draw velocity arrow on top
            arcade.draw_line(x, y, x + vX, y + vY, AGENT_ARROW_COLOR, 4)

            self.data.append(
                {
                    "lidar": [l.model_dump() for l in agent.lidar_observation],
                    "action": [vX, vY],
                    "reward": agent.last_reward,
                }
            )

        # Capture frame for recording
        if self.record:
            self._capture_frame()

    def _draw_agent_rays(self, agent: AgentState):
        """
        Draw rays for a single agent with different colors based on what they hit.
        """
        import math

        total_rays = len(agent.lidar_observation)
        if total_rays == 0:
            return

        fov_degrees = agent.fov_degrees
        max_range = agent.max_range

        # Get agent's current facing direction from explicit direction field
        agent_facing_angle = math.atan2(agent.direction[1], agent.direction[0])

        # Calculate ray directions based on FOV relative to agent's facing direction
        start_angle = agent_facing_angle - math.radians(fov_degrees / 2)
        angle_increment = (
            math.radians(fov_degrees) / (total_rays - 1) if total_rays > 1 else 0
        )

        for i, ray_result in enumerate(agent.lidar_observation):
            # Calculate ray angle and direction relative to agent's facing direction
            ray_angle_rad = start_angle + i * angle_increment

            # Ray direction accounting for agent's facing direction
            ray_dir_x = math.cos(ray_angle_rad)
            ray_dir_y = math.sin(ray_angle_rad)

            # Scale ray origin to screen coordinates
            start_x = agent.position[0] * self.width
            start_y = agent.position[1] * self.height

            alpha = RAY_ALPHA
            line_width = RAY_LINE_WIDTH

            # Determine ray color based on what it hit
            if ray_result.intersecting_with == "obstacle":
                ray_color = RAY_OBSTACLE_HIT_COLOR
            elif ray_result.intersecting_with == "boundary":
                ray_color = RAY_BOUNDARY_HIT_COLOR
            elif ray_result.intersecting_with == "goal":
                ray_color = RAY_GOAL_HIT_COLOR
            elif ray_result.intersecting_with == "agent":
                ray_color = RAY_AGENT_HIT_COLOR
            else:
                ray_color = RAY_MISS_COLOR

            # Calculate end point
            if ray_result.intersection:
                # Ray hit something - draw to intersection point
                end_x = ray_result.intersection.x * self.width
                end_y = ray_result.intersection.y * self.height
            else:
                # Ray missed - draw full length
                end_x = start_x + ray_dir_x * max_range * self.width
                end_y = start_y + ray_dir_y * max_range * self.height

            color_with_alpha = (*ray_color[:3], alpha)
            arcade.draw_line(
                start_x, start_y, end_x, end_y, color_with_alpha, line_width
            )

            # # # Draw intersection marker
            if DRAW_INTERSECTION_MARKERS:
                if ray_result.intersection:
                    marker_x = ray_result.intersection.x * self.width
                    marker_y = ray_result.intersection.y * self.height

                    marker_size = 2
                    arcade.draw_circle_filled(
                        marker_x,
                        marker_y,
                        marker_size,
                        RAY_INTERSECTION_MARKER_COLOR,
                        1,
                    )

    def _capture_frame(self):
        """
        Capture the current frame for recording.
        """
        if not self.record or not IMAGEIO_AVAILABLE:
            return

        # Get the current frame as RGB data
        image = arcade.get_image()
        # Convert PIL image to numpy array
        frame = np.array(image)
        # Convert RGBA to RGB if necessary
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]

        # Add frame to collection
        self.frames.append(frame)

        # # If we have too many frames in memory, start writing to file
        # if len(self.frames) > 120:  # About 4 seconds at 30 FPS
        #     self._flush_frames_to_video()

    def _flush_frames_to_video(self):
        """
        Write accumulated frames to video file.
        """
        if not self.frames:
            return

        # json.dump(self.data, open("movies/data/episode2.json", "w"))
        # all_frames = np.array(self.frames)
        # np.save("movies/data/frames2.npy", all_frames)

        # Initialize video writer if not already done
        if self.recording_writer is None:
            self.recording_writer = imageio.get_writer(
                "movies/footage.mp4",
                fps=self.target_fps,
                codec="libx264",
                quality=8,
                pixelformat="yuv420p",
            )
            print("🎬 Started writing video to movies/footage.mp4")

        # Write all accumulated frames
        for frame in self.frames:
            self.recording_writer.append_data(frame)

        # Clear frames from memory
        self.frames.clear()

    def stop_recording(self):
        """
        Stop recording and save the final video file.
        """
        if not self.record:
            return

        try:
            # Write any remaining frames
            self._flush_frames_to_video()

            # Close the video writer
            if self.recording_writer is not None:
                self.recording_writer.close()
                self.recording_writer = None
                print("🎬 Recording saved to movies/footage.mp4")

        except Exception as e:
            print(f"Warning: Failed to finalize recording: {e}")

    def on_close(self):
        """
        Handle window close event - save recording if active.
        """
        self.stop_recording()
        super().on_close()
