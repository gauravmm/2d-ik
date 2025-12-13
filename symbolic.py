#!python3


from typing import Any

from datamodel import Position, RobotModel, RobotPosition, RobotState


class IKSymbolic:
    """Implements a symbolic solver for inverse kinematics using the Sympy solver."""

    def __init__(self, model: RobotModel) -> None:
        # Create variables for each joint rotation and set up a system of equations that
        # relate each end effector position
        self.model = model

    def __call__(self, state: RobotState) -> RobotPosition:
        # Sanity-check that state.model is the same as self.model
        # Given the desired end effector position:
        state.desired_end_effector
        # Use the sympy equations prepared in init to find joint angles that achieve
        # the requested position. If the position is not possible, find the closest
        # legal position.
        pass
