#!python3

import math
from dataclasses import dataclass, field
from typing import Collection, List, Optional, Tuple

#
# Robot Model
#


@dataclass(frozen=True)
class RobotModel:
    """Defines the parameters for a robot on a 2d plane consisting of a chain of rigid links with revolute joints between them. Link displacements and joint origins must be configurable."""

    link_lengths: tuple[float, ...]  # Length of each link in the chain
    joint_origins: tuple[float, ...] = field(default=tuple())  # Initial angle offset
    joint_limits: tuple[tuple[float, float] | None, ...] = field(
        default=tuple()
    )  # Joint angle limits, set in joint space. These joint angle limits are relative to the joint origin

    def __post_init__(self):
        # Validate that we have consistent dimensions
        if self.joint_origins and len(self.joint_origins) != len(self.link_lengths):
            raise ValueError(
                f"Number of joint origins ({len(self.joint_origins)}) must match number of links ({len(self.link_lengths)})"
            )

        # If no joint origins specified, set them all to 0
        if not self.joint_origins:
            object.__setattr__(
                self, "joint_origins", tuple(0.0 for _ in self.link_lengths)
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


@dataclass(frozen=True)
class RegionBall:
    """Ball (circle) region defined by points within radius r from center.

    Points inside the ball satisfy: ||point - center|| <= radius
    """

    center: Tuple[float, float]  # Center of the ball
    radius: float  # Radius of the ball


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


Region = RegionHalfspace | RegionBall | RegionRectangle


@dataclass(frozen=True)
class WorldModel:
    nogo: Collection[Region] = field(default=tuple())


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
