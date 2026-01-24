#!python3

import time
from typing import Tuple

import sympy as sp
import sympy.solvers as solvers

from datamodel import (
    DesiredPosition,
    IKReturn,
    IKSolver,
    RobotModel,
    RobotPosition,
    RobotState,
    WorldModel,
)


class IKSymbolic(IKSolver):
    """Implements a symbolic solver for inverse kinematics using the Sympy solver."""

    def __init__(self, model: RobotModel, world: WorldModel | None) -> None:
        # Create variables for each joint rotation and set up a system of equations that
        # relate each end effector position
        self.model = model
        self.nogo = world and world.nogo
        self.n_joints = len(model.link_lengths)
        assert self.n_joints == 3, "Symbolic solver requires exactly 3 joints."

        # Process joint limits from model
        # joint_limits is a tuple of (min, max) or None for each joint
        self.joint_bounds: list[tuple[float, float] | None] = []
        if model.joint_limits:
            for i, limit in enumerate(model.joint_limits):
                if limit is not None:
                    self.joint_bounds.append((limit[0], limit[1]))
                else:
                    self.joint_bounds.append(None)
        # Pad with None if joint_limits is shorter than number of joints
        while len(self.joint_bounds) < self.n_joints:
            self.joint_bounds.append(None)

        # Create symbolic variables for joint angles
        thetas = sp.symbols(f"theta0:{self.n_joints}", real=True, seq=True)
        assert isinstance(thetas, tuple)
        self.theta_symbols: Tuple[sp.Symbol, ...] = thetas

        # Build symbolic forward kinematics equations
        # Store positions and angles at each joint for boundary checking
        joint_x_syms: list[sp.Expr] = []
        joint_y_syms: list[sp.Expr] = []
        joint_angles_syms: list[sp.Expr] = []

        x_sym: sp.Expr = sp.Float(0)
        y_sym: sp.Expr = sp.Float(0)
        cumulative_angle = sp.Float(0)

        for i, (link_length, joint_origin) in enumerate(
            zip(model.link_lengths, model.joint_origins)
        ):
            # Add joint angle and origin to cumulative angle
            cumulative_angle = cumulative_angle + self.theta_symbols[i] + joint_origin

            # Add this link's contribution to end effector position
            x_sym = x_sym + link_length * sp.cos(cumulative_angle)
            y_sym = y_sym + link_length * sp.sin(cumulative_angle)

            # Store position and angle at this joint
            joint_x_syms.append(x_sym)
            joint_y_syms.append(y_sym)
            joint_angles_syms.append(cumulative_angle)

        # Store the symbolic end effector position equations
        self.end_effector_x: sp.Expr = x_sym
        self.end_effector_y: sp.Expr = y_sym

        # Store the symbolic end effector orientation
        self.end_effector_angle: sp.Expr = cumulative_angle

        # Store symbolic positions and angles at each joint for boundary checking
        self.joint_x_syms: Tuple[sp.Expr, ...] = tuple(joint_x_syms)
        self.joint_y_syms: Tuple[sp.Expr, ...] = tuple(joint_y_syms)
        self.joint_angles_syms: Tuple[sp.Expr, ...] = tuple(joint_angles_syms)

        if self.nogo:
            print("WARNING: Nogo zones not supported for symbolic solver.")

        # Create symbolic variables for desired end effector position and angle
        self.target_x_sym = sp.Symbol("target_x", real=True)
        self.target_y_sym = sp.Symbol("target_y", real=True)
        self.target_angle_sym = sp.Symbol("target_angle", real=True)

        # Set up equations: end_effector_expr = target_value
        equations = [
            sp.Eq(self.end_effector_x, self.target_x_sym),
            sp.Eq(self.end_effector_y, self.target_y_sym),
            sp.Eq(self.end_effector_angle, self.target_angle_sym),
        ]

        # Solve for theta symbols in terms of target_x, target_y, target_angle
        self.solved = solvers.solve(
            equations,
            self.theta_symbols,
            dict=True,
        )
        print(f"Symbolic solutions found: {len(self.solved)}")
        for i, sol in enumerate(self.solved):
            print(f"  Solution {i}: {sol}")

    def __call__(
        self,
        state: RobotState,
        desired: DesiredPosition,
    ) -> IKReturn:
        """Solve IK for the desired position.

        Args:
            state: Current robot state.
            desired: Desired end effector position and optional angle.

        Returns:
            IKReturn containing the solution state and profiling information.
        """
        # Sanity-check that state.model is the same as self.model
        if state.model != self.model:
            raise ValueError("State model does not match IKSymbolic model")

        # Get the desired end effector position
        if desired.ee_position is None:
            raise ValueError("DesiredPosition must have an ee_position")

        target_x, target_y = desired.ee_position
        # Copy the previous final angle if not specified.
        target_angle = desired.ee_angle
        if target_angle is None:
            target_angle = 0.0
            print("WARNING: Target angle not specified.")

        start_time = time.perf_counter()

        # Use precomputed symbolic solutions
        if not self.solved:
            raise ValueError("No symbolic solution available")

        # Substitution dict for target values
        subs_dict = {
            self.target_x_sym: target_x,
            self.target_y_sym: target_y,
            self.target_angle_sym: target_angle,
        }

        # Evaluate all solutions and pick the best one (closest to current position)
        best_solution = None
        best_distance = float("inf")

        for sol in self.solved:
            try:
                # Substitute target values into the symbolic solution
                joint_angles = tuple(
                    float(sol[t].subs(subs_dict).evalf()) for t in self.theta_symbols
                )

                # Calculate distance from current joint angles
                distance = sum(
                    (a - b) ** 2
                    for a, b in zip(joint_angles, state.current.joint_angles)
                )

                if distance < best_distance:
                    best_distance = distance
                    best_solution = joint_angles
            except (TypeError, ValueError):
                # Solution may be complex or undefined for these target values
                continue

        end_time = time.perf_counter()

        if best_solution is None:
            raise ValueError("No valid solution found for target position")

        joint_angles = best_solution
        # Extract joint angles from solution
        # TODO: Enforce joint angles.

        result_state = state.with_position(
            RobotPosition(joint_angles=joint_angles), desired=desired
        )

        # Compute position error
        positions = self.model.forward_kinematics(
            RobotPosition(joint_angles=joint_angles)
        )
        ee_pos = positions[-1]
        position_error = (
            (ee_pos[0] - target_x) ** 2 + (ee_pos[1] - target_y) ** 2
        ) ** 0.5

        return IKReturn(
            state=result_state,
            solve_time_ms=(end_time - start_time) * 1000,
            iterations=1,
            converged=position_error < 1e-2,
            initial_loss=-1,
            final_loss=position_error,
            position_error=position_error,
        )
