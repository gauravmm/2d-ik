from matplotlib import artist
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.lines import Line2D
from matplotlib.backend_bases import MouseEvent, MouseButton, Event
import matplotlib.animation as animation
from datamodel import (
    DesiredPosition,
    RegionBall,
    RegionHalfspace,
    RegionRectangle,
    RobotModel,
    RobotPosition,
    RobotState,
)
from typing import Callable, List, Literal, Optional, Sequence
import numpy as np
import logging

logger = logging.getLogger(__name__)

ClickCallbackType = Callable[[float, float, Literal["left", "right"]], None]


class RobotVisualizer:
    """Interactive visualization for a 2D robot arm with scale drawing and click event handling."""

    def __init__(
        self,
        robot_state: RobotState,
        figsize: tuple[float, float] = (10, 10),
        click_callback: Optional[ClickCallbackType] = None,
        joint_radius: float = 0.05,
        link_width: float = 2.0,
        auto_scale: bool = True,
        scale_margin: float = 1.2,
    ):
        """Initialize the robot visualizer.

        Args:
            robot_state: Initial RobotPosition to display
            figsize: Figure size in inches (width, height)
            click_callback: Optional callback function that receives (x, y) coordinates on click
            joint_radius: Radius of joint circles (in world coordinates)
            link_width: Width of link lines (in points)
            auto_scale: Whether to automatically scale the view based on robot reach
            scale_margin: Margin multiplier for auto-scaling (e.g., 1.2 = 20% margin)
        """
        self.robot_state = robot_state
        self.click_callback = click_callback
        self.joint_radius = joint_radius
        self.link_width = link_width
        self.auto_scale = auto_scale
        self.scale_margin = scale_margin

        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel("X (units)")
        self.ax.set_ylabel("Y (units)")
        self.ax.set_title("2D Robot Arm Visualization")

        # Initialize plot elements
        self.link_lines: list[Line2D] = []
        self.joint_circles: list[Circle] = []
        self.end_effector_circle: Optional[Circle] = None

        # Set up click event handling
        if self.click_callback:
            self._click_handler_id = self.fig.canvas.mpl_connect(
                "button_press_event", self._on_click
            )

        # Initial draw
        self._setup_plot_elements()
        self._update_view_limits()
        self.update(self.robot_state)

    def _render_world_regions(self):
        """Render the nogo regions from the WorldModel as pink shapes."""
        world = self.robot_state.world
        if not world.nogo:
            return

        # Get the view bounds for rendering halfspaces
        max_reach = sum(self.robot_state.model.link_lengths)
        bound = max_reach * self.scale_margin * 2

        for region in world.nogo:
            if isinstance(region, RegionBall):
                # Render as a filled circle
                circle = Circle(
                    region.center,
                    region.radius,
                    color="pink",
                    fill=True,
                    alpha=0.7,
                    zorder=0,
                )
                self.ax.add_patch(circle)

            elif isinstance(region, RegionRectangle):
                # Render as a filled rectangle
                rect = Rectangle(
                    (region.left, region.bottom),
                    region.right - region.left,
                    region.top - region.bottom,
                    color="pink",
                    fill=True,
                    alpha=0.7,
                    zorder=0,
                )
                self.ax.add_patch(rect)

            elif isinstance(region, RegionHalfspace):
                # Render as a large polygon covering the halfspace
                # The halfspace is defined as: normal · (point - anchor) >= 0
                # Points satisfying this are "inside" the halfspace (the nogo region)
                nx, ny = region.normal
                ax, ay = region.anchor

                # Normalize the normal vector
                norm_len = np.sqrt(nx * nx + ny * ny)
                if norm_len < 1e-10:
                    continue
                nx, ny = nx / norm_len, ny / norm_len

                # Direction along the boundary (perpendicular to normal)
                tx, ty = -ny, nx

                # Create a large polygon representing the halfspace
                # Start from anchor and extend in both directions along boundary
                # Then extend "into" the halfspace (in the direction of the normal)
                p1 = (ax - tx * bound, ay - ty * bound)
                p2 = (ax + tx * bound, ay + ty * bound)
                p3 = (ax + tx * bound + nx * bound, ay + ty * bound + ny * bound)
                p4 = (ax - tx * bound + nx * bound, ay - ty * bound + ny * bound)

                poly = Polygon(
                    [p1, p2, p3, p4],
                    color="pink",
                    fill=True,
                    alpha=0.7,
                    zorder=0,
                )
                self.ax.add_patch(poly)

    def _setup_plot_elements(self):
        """Create the plot elements for links and joints."""
        # Render world regions first (so they appear behind the robot)
        self._render_world_regions()

        num_links = len(self.robot_state.model.link_lengths)

        # Create link lines
        for i in range(num_links):
            line = Line2D(
                [0, 0],
                [0, 0],
                linewidth=self.link_width,
                color="blue",
                solid_capstyle="round",
                zorder=1,
            )
            self.ax.add_line(line)
            self.link_lines.append(line)

        # Create joint circles (including base and end)
        for i in range(num_links + 1):
            circle = Circle((0, 0), self.joint_radius, color="red", fill=True, zorder=2)
            self.ax.add_patch(circle)
            self.joint_circles.append(circle)

        # Create end effector circle (different color)
        self.end_effector_circle = Circle(
            (0, 0), self.joint_radius * 1.2, color="green", fill=True, zorder=3
        )
        self.ax.add_patch(self.end_effector_circle)

    def _update_view_limits(self):
        """Update the view limits based on robot configuration."""
        if self.auto_scale:
            # Calculate maximum reach (sum of all link lengths)
            max_reach = sum(self.robot_state.model.link_lengths)
            margin = max_reach * self.scale_margin

            self.ax.set_xlim(-margin, margin)
            self.ax.set_ylim(-margin, margin)

    def _on_click(self, event: Event):
        """Handle mouse click events."""
        assert isinstance(event, MouseEvent)
        if (
            event.inaxes == self.ax
            and event.xdata is not None
            and event.ydata is not None
        ):
            if self.click_callback:
                self.click_callback(
                    event.xdata,
                    event.ydata,
                    "left" if event.button == MouseButton.LEFT else "right",
                )
            else:
                logger.warning("No callback registered!")

    def update(self, robot_state: RobotState):
        """Update the visualization with a new robot position.

        Args:
            robot_state: New RobotState to display
        """
        self.robot_state = robot_state

        # Get joint positions
        positions = robot_state.get_joint_positions()

        # Update link lines
        for line, fr, to in zip(self.link_lines, positions, positions[1:]):
            x_data = [fr[0], to[0]]
            y_data = [fr[1], to[1]]
            line.set_data(x_data, y_data)

        for circle, pos in zip(self.joint_circles, positions):
            circle.center = pos

        # Update end effector
        if self.end_effector_circle:
            if (
                robot_state.desired is not None
                and robot_state.desired.ee_position is not None
            ):
                self.end_effector_circle.center = robot_state.desired.ee_position
            else:
                self.end_effector_circle.center = (999.0, 999.0)

        # Force redraw
        if hasattr(self.fig.canvas, "draw_idle"):
            self.fig.canvas.draw_idle()
        else:
            self.fig.canvas.draw()

    def show(self):
        """Display the visualization window."""
        plt.show()

    def animate(
        self,
        update_func: Callable[[int], RobotState],
        interval: int = 50,
        frames: Optional[int] = None,
    ):
        """Run an animation loop that updates the robot position.

        Args:
            update_func: Function that takes frame number and returns a RobotPosition
            interval: Time between frames in milliseconds
            frames: Number of frames (None for infinite loop)
        """

        def animate_frame(frame) -> Sequence[artist.Artist]:
            new_state = update_func(frame)
            self.update(new_state)
            # This construction is to make the typechecker happy:
            rv: List[artist.Artist] = [] + self.link_lines + self.joint_circles
            if self.end_effector_circle:
                rv.append(self.end_effector_circle)
            return rv

        anim = animation.FuncAnimation(
            self.fig, animate_frame, frames=frames, interval=interval, blit=False
        )

        plt.show()
        return anim

    def set_click_callback(self, callback: ClickCallbackType):
        """Set or update the click event callback.

        Args:
            callback: Function that receives (x, y) coordinates on click
        """
        self.click_callback = callback
        # Ensure the event handler is connected
        if not hasattr(self, "_click_handler_id"):
            self._click_handler_id = self.fig.canvas.mpl_connect(
                "button_press_event", self._on_click
            )


# Example usage
if __name__ == "__main__":
    import math

    # Create a simple 3-link robot
    model = RobotModel(link_lengths=(1.0, 0.8, 0.6), joint_origins=(0.0, 0.0, 0.0))

    # Initial position
    position = RobotPosition(joint_angles=(0.0, math.pi / 4, -math.pi / 4))
    state = RobotState(model, position, desired=DesiredPosition(ee_position=(0, 0)))

    # Click handler
    def on_click(x, y, btn):
        print(f"Clicked at: ({x:.2f}, {y:.2f}) {btn}")

    # Create visualizer
    viz = RobotVisualizer(state, click_callback=on_click)

    # Example: Update loop with animation
    global frame_count
    frame_count = 0

    def update_robot(frame):
        global frame_count
        frame_count = frame
        # Animate by rotating the first joint
        new_angles = (
            math.sin(frame * 0.05),
            math.pi / 4 + math.cos(frame * 0.03),
            -math.pi / 4 + math.sin(frame * 0.07),
        )
        position = RobotPosition(joint_angles=new_angles)
        return RobotState(
            model,
            position,
            desired=DesiredPosition(
                ee_position=(math.sin(frame * 0.04), math.cos(frame * 0.04))
            ),
        )

    # Run animation
    viz.animate(update_robot, interval=50)
