#!python3

from dataclasses import dataclass, field
import math
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class RobotModel:
    """Defines the parameters for a robot on a 2d plane consisting of a chain of rigid links with revolute joints between them. Link displacements and joint origins must be configurable."""

    link_lengths: tuple[float, ...]  # Length of each link in the chain
    joint_origins: tuple[float, ...] = field(default=tuple())  # Initial angle offset

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
class RobotState:
    """Encodes the state of the robot at this point in time."""

    model: RobotModel
    current: RobotPosition
    desired_end_effector: Optional[Position] = None
    desired_end_effector_angle: Optional[float] = None

    def get_joint_positions(self) -> List[Position]:
        """Convenience method to calculate joint positions using forward kinematics.

        Returns:
            A list of (x, y) tuples representing the position of each joint.
        """
        return self.model.forward_kinematics(self.current)
