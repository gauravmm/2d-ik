#!/usr/bin/env python3

from matplotlib import artist
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import matplotlib.animation as animation
from datamodel import RobotModel, RobotPosition, RobotState
from typing import Callable, Iterable, List, Optional, Sequence


class RobotVisualizer:
    """Interactive visualization for a 2D robot arm with scale drawing and click event handling."""

    def __init__(
        self,
        robot_state: RobotState,
        figsize: tuple[float, float] = (10, 10),
        click_callback: Optional[Callable[[float, float], None]] = None,
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
            self._click_handler_id = self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        # Initial draw
        self._setup_plot_elements()
        self._update_view_limits()
        self.update(self.robot_state)

    def _setup_plot_elements(self):
        """Create the plot elements for links and joints."""
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

        # Create joint circles (including base)
        for i in range(num_links):
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

    def _on_click(self, event):
        """Handle mouse click events."""
        # Debug: Print when any click is detected
        import sys
        print(f"[DEBUG] Click event received: inaxes={event.inaxes == self.ax}, x={event.xdata}, y={event.ydata}", file=sys.stderr, flush=True)

        if (
            event.inaxes == self.ax
            and event.xdata is not None
            and event.ydata is not None
        ):
            if self.click_callback:
                self.click_callback(event.xdata, event.ydata)
            else:
                print(f"[DEBUG] No callback registered!", file=sys.stderr, flush=True)

    def update(self, robot_state: RobotState):
        """Update the visualization with a new robot position.

        Args:
            robot_state: New RobotState to display
        """
        self.robot_state = robot_state

        # Get joint positions
        positions = robot_state.get_joint_positions()

        # Update link lines
        for i, line in enumerate(self.link_lines):
            x_data = [positions[i][0], positions[i + 1][0]]
            y_data = [positions[i][1], positions[i + 1][1]]
            line.set_data(x_data, y_data)

        # Update joint circles (skip the last position, that's the end effector)
        for i, circle in enumerate(self.joint_circles):
            circle.center = positions[i]

        # Update end effector
        if self.end_effector_circle:
            if robot_state.desired_end_effector is not None:
                self.end_effector_circle.center = robot_state.desired_end_effector
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

    def set_click_callback(self, callback: Callable[[float, float], None]):
        """Set or update the click event callback.

        Args:
            callback: Function that receives (x, y) coordinates on click
        """
        self.click_callback = callback
        # Ensure the event handler is connected
        if not hasattr(self, '_click_handler_id'):
            self._click_handler_id = self.fig.canvas.mpl_connect("button_press_event", self._on_click)


# Example usage
if __name__ == "__main__":
    import math

    # Create a simple 3-link robot
    model = RobotModel(link_lengths=(1.0, 0.8, 0.6), joint_origins=(0.0, 0.0, 0.0))

    # Initial position
    position = RobotPosition(joint_angles=(0.0, math.pi / 4, -math.pi / 4))
    state = RobotState(model, position, (0, 0))

    # Click handler
    def on_click(x, y):
        print(f"Clicked at: ({x:.2f}, {y:.2f})")

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
            model, position, (math.sin(frame * 0.04), math.cos(frame * 0.04))
        )

    # Run animation
    viz.animate(update_robot, interval=50)
