#!python3

import sympy as sp
from typing import Any

from datamodel import Position, RobotModel, RobotPosition, RobotState


class IKSymbolic:
    """Implements a symbolic solver for inverse kinematics using the Sympy solver."""

    def __init__(self, model: RobotModel) -> None:
        # Create variables for each joint rotation and set up a system of equations that
        # relate each end effector position
        self.model = model
        self.n_joints = len(model.link_lengths)

        # Create symbolic variables for joint angles
        self.theta_symbols = sp.symbols(f'theta0:{self.n_joints}', real=True)

        # Build symbolic forward kinematics equations
        x_sym = sp.Float(0)
        y_sym = sp.Float(0)
        cumulative_angle = sp.Float(0)

        for i, (link_length, joint_origin) in enumerate(zip(model.link_lengths, model.joint_origins)):
            # Add joint angle and origin to cumulative angle
            cumulative_angle = cumulative_angle + self.theta_symbols[i] + joint_origin

            # Add this link's contribution to end effector position
            x_sym = x_sym + link_length * sp.cos(cumulative_angle)
            y_sym = y_sym + link_length * sp.sin(cumulative_angle)

        # Store the symbolic end effector position equations
        self.end_effector_x = x_sym
        self.end_effector_y = y_sym

    def __call__(self, state: RobotState) -> RobotPosition:
        # Sanity-check that state.model is the same as self.model
        if state.model != self.model:
            raise ValueError("State model does not match IKSymbolic model")

        # Get the desired end effector position
        if state.desired_end_effector is None:
            raise ValueError("State must have a desired_end_effector position")

        target_x, target_y = state.desired_end_effector

        # For inverse kinematics, we want to minimize the distance to target
        # Create distance squared function
        distance_squared = (self.end_effector_x - target_x)**2 + (self.end_effector_y - target_y)**2

        # Convert to a numerical function
        distance_func = sp.lambdify(self.theta_symbols, distance_squared, 'numpy')

        # Use scipy for optimization
        from scipy.optimize import minimize

        # Initial guess from current state
        x0 = list(state.current.joint_angles)

        # Minimize distance to target
        result = minimize(
            lambda x: distance_func(*x),
            x0,
            method='BFGS'
        )

        # Extract joint angles from solution
        joint_angles = tuple(float(angle) for angle in result.x)

        return RobotPosition(joint_angles=joint_angles)
