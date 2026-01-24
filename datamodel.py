#!python3

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

#
# Robot Model
#


@dataclass(frozen=True)
class RobotModel:
    """Defines the parameters for a robot on a 2d plane consisting of a chain of rigid links with revolute joints between them. Link displacements and joint origins must be configurable."""

    link_lengths: tuple[float, ...]  # Length of each link in the chain
    joint_origins: tuple[float, ...] = field(default=tuple())  # Initial angle offset
    joint_limits: tuple[tuple[float, float], ...] = field(
        default=tuple()
    )  # Joint angle limits, set in joint space. These joint angle limits are relative to the joint origin

    def __post_init__(self):
        # Validate that we have consistent dimensions
        if self.joint_origins and len(self.joint_origins) != len(self.link_lengths):
            raise ValueError(
                f"Number of joint origins ({len(self.joint_origins)}) must match number of links ({len(self.link_lengths)})"
            )
        if self.joint_limits and len(self.joint_limits) != len(self.link_lengths):
            raise ValueError(
                f"Number of joint limits ({len(self.joint_limits)}) must match number of links ({len(self.link_lengths)})"
            )

        # If no joint origins specified, set them all to 0
        if not self.joint_origins:
            object.__setattr__(
                self, "joint_origins", tuple(0.0 for _ in self.link_lengths)
            )

        # If no joint limits specified, set them all to (-inf, +inf)
        if not self.joint_limits:
            object.__setattr__(
                self,
                "joint_limits",
                tuple((-math.inf, math.inf) for _ in self.link_lengths),
            )

    def forward_kinematics(self, position: "RobotPosition") -> List["Position"]:
        """Calculate the (x, y) position of each joint in the chain using forward kinematics.

        Args:
            position: The RobotPosition containing joint angles

        Returns:
            A list of (x, y) tuples representing the position of each joint,
            starting with the base at (0, 0) and ending with the end effector position.
        """
        positions = [(0.0, 0.0)]  # Base position at origin
        current_x, current_y = 0.0, 0.0
        cumulative_angle = 0.0

        for link_length, joint_angle, joint_origin in zip(
            self.link_lengths, position.joint_angles, self.joint_origins
        ):
            # Add the joint angle (relative) and joint origin (offset) to cumulative angle
            cumulative_angle += joint_angle + joint_origin

            # Calculate the end position of this link
            current_x += link_length * math.cos(cumulative_angle)
            current_y += link_length * math.sin(cumulative_angle)

            positions.append((current_x, current_y))

        return positions


Position = Tuple[float, float]


@dataclass(frozen=True)
class RobotPosition:
    """Defines the current joint rotation values of the 2d robot, relative to the joint origin defined in RobotModel."""

    joint_angles: tuple[float, ...]  # Current rotation angle for each joint (radians)


@dataclass(frozen=True)
class DesiredPosition:
    ee_position: Position
    ee_angle: Optional[float] = None


#
# World Model
#


# These regions are used to limit the robot's access. These must all be convex and have a closed-form computation.
# Each Region returns a residual error, where positive numbers indicate the point lies inside the region and negative
# numbers indicate the point is outside the region.
@dataclass(frozen=True)
class RegionHalfspace:
    """Halfspace region defined by: normal · (point - anchor) >= 0

    Points inside the halfspace satisfy the inequality.
    The normal vector points toward the interior of the halfspace.
    """

    normal: Tuple[float, float]  # Normal vector pointing inward
    anchor: Tuple[float, float]  # Point on the boundary plane

    def point(self, p: Tuple[float, float]) -> bool:
        """Check if a point collides with (is inside) this halfspace."""
        dx = p[0] - self.anchor[0]
        dy = p[1] - self.anchor[1]
        return self.normal[0] * dx + self.normal[1] * dy >= 0

    def line(self, segment: Tuple[Tuple[float, float], Tuple[float, float]]) -> bool:
        """Check if a line segment collides with (intersects or is inside) this halfspace."""
        p1, p2 = segment
        # If either endpoint is inside, there's a collision
        return self.point(p1) or self.point(p2)


@dataclass(frozen=True)
class RegionBall:
    """Ball (circle) region defined by points within radius r from center.

    Points inside the ball satisfy: ||point - center|| <= radius
    """

    center: Tuple[float, float]  # Center of the ball
    radius: float  # Radius of the ball

    def point(self, p: Tuple[float, float]) -> bool:
        """Check if a point collides with (is inside) this ball."""
        dx = p[0] - self.center[0]
        dy = p[1] - self.center[1]
        return dx * dx + dy * dy <= self.radius * self.radius

    def line(self, segment: Tuple[Tuple[float, float], Tuple[float, float]]) -> bool:
        """Check if a line segment collides with (intersects or is inside) this ball."""
        p1, p2 = segment
        # Check endpoints first
        if self.point(p1) or self.point(p2):
            return True

        # Find closest point on segment to center
        seg_dx = p2[0] - p1[0]
        seg_dy = p2[1] - p1[1]
        seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy

        if seg_len_sq == 0:
            # Degenerate segment (point)
            return self.point(p1)

        # Project center onto line, clamped to segment [0, 1]
        to_center_x = self.center[0] - p1[0]
        to_center_y = self.center[1] - p1[1]
        t = (to_center_x * seg_dx + to_center_y * seg_dy) / seg_len_sq
        t = max(0.0, min(1.0, t))

        # Closest point on segment
        closest_x = p1[0] + t * seg_dx
        closest_y = p1[1] + t * seg_dy

        return self.point((closest_x, closest_y))


@dataclass(frozen=True)
class RegionRectangle:
    """Axis-aligned rectangle region defined by left, right, bottom, and top boundaries.

    Points inside the rectangle satisfy: left <= x <= right and bottom <= y <= top
    """

    left: float  # Left boundary (minimum x)
    right: float  # Right boundary (maximum x)
    bottom: float  # Bottom boundary (minimum y)
    top: float  # Top boundary (maximum y)

    def __post_init__(self):
        """Validate that boundaries are correctly ordered."""
        if self.left >= self.right:
            raise ValueError(
                f"left ({self.left}) must be less than right ({self.right})"
            )
        if self.bottom >= self.top:
            raise ValueError(
                f"bottom ({self.bottom}) must be less than top ({self.top})"
            )

    def point(self, p: Tuple[float, float]) -> bool:
        """Check if a point collides with (is inside) this rectangle."""
        return self.left <= p[0] <= self.right and self.bottom <= p[1] <= self.top

    def line(self, segment: Tuple[Tuple[float, float], Tuple[float, float]]) -> bool:
        """Check if a line segment collides with (intersects or is inside) this rectangle."""
        p1, p2 = segment
        # Check endpoints first
        if self.point(p1) or self.point(p2):
            return True

        # Check if segment intersects any of the four edges
        # Using parametric line-segment intersection
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        # Check intersection with each edge
        def intersects_vertical_edge(edge_x: float) -> bool:
            if dx == 0:
                return False
            t = (edge_x - p1[0]) / dx
            if 0 <= t <= 1:
                y_at_edge = p1[1] + t * dy
                return self.bottom <= y_at_edge <= self.top
            return False

        def intersects_horizontal_edge(edge_y: float) -> bool:
            if dy == 0:
                return False
            t = (edge_y - p1[1]) / dy
            if 0 <= t <= 1:
                x_at_edge = p1[0] + t * dx
                return self.left <= x_at_edge <= self.right
            return False

        return (
            intersects_vertical_edge(self.left)
            or intersects_vertical_edge(self.right)
            or intersects_horizontal_edge(self.bottom)
            or intersects_horizontal_edge(self.top)
        )


Region = RegionHalfspace | RegionBall | RegionRectangle


@dataclass(frozen=True)
class WorldModel:
    nogo: Tuple[Region, ...] = field(default=tuple())


#
# Overall state
#


@dataclass(frozen=True)
class RobotState:
    """Encodes the state of the robot at this point in time."""

    model: RobotModel
    current: RobotPosition
    world: WorldModel = field(default=WorldModel())
    desired: Optional[DesiredPosition] = field(default=None)

    def get_joint_positions(self) -> List[Position]:
        """Convenience method to calculate joint positions using forward kinematics.

        Returns:
            A list of (x, y) tuples representing the position of each joint.
        """
        return self.model.forward_kinematics(self.current)

    def with_position(
        self, position: RobotPosition, desired: DesiredPosition | None = None
    ):
        return RobotState(
            self.model,
            current=position,
            world=self.world,
            desired=desired or self.desired or None,
        )


@dataclass
class IKReturn:
    """Unified return type for all IK solvers, containing the solution state and profiling information."""

    state: RobotState  # The solved robot state
    solve_time_ms: float  # Total solve time in milliseconds
    iterations: int  # Number of optimization iterations
    converged: bool  # Whether the solver converged before max_iterations
    initial_loss: float  # Loss at the start of optimization
    final_loss: float  # Loss at the end of optimization
    position_error: float  # Final Euclidean distance to target position

    # TODO: Remove initial_loss.


class IKSolver(ABC):
    """Base interface for all IK solvers."""

    @abstractmethod
    def __call__(self, state: RobotState, desired: DesiredPosition) -> IKReturn:
        """Solve IK for the desired position.

        Args:
            state: Current robot state.
            desired: Desired end effector position and optional angle.

        Returns:
            IKReturn containing the solution state and profiling information.
        """
        ...
