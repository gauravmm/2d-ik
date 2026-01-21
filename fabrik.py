#!python3

"""FABRIK (Forward And Backward Reaching Inverse Kinematics) solver.

FABRIK is a heuristic iterative method that solves IK by alternating between:
1. Forward reaching: Move end effector to target, then adjust joints backward to base
2. Backward reaching: Fix base position, then adjust joints forward to end effector

This implementation follows the interface pattern from symbolic.py but uses
numpy arrays for numerical operations.
"""

import math
import time
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

from datamodel import (
    DesiredPosition,
    RobotModel,
    RobotPosition,
    RobotState,
    WorldModel,
)


@dataclass
class IKFabrikProfile:
    """Profiling results from IKFabrik solver."""

    solve_time_ms: float  # Total solve time in milliseconds
    iterations: int  # Number of FABRIK iterations
    converged: bool  # Whether the solver converged before max_iterations
    initial_error: float  # Position error at the start
    final_error: float  # Position error at the end
    position_error: float  # Final Euclidean distance to target position


class IKFabrik:
    """Implements FABRIK inverse kinematics solver using numpy."""

    def __init__(
        self,
        model: RobotModel,
        world: WorldModel | None,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> None:
        self.model = model
        self.n_joints = len(model.link_lengths)

        # Optimization parameters
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Store link lengths as numpy array
        self.link_lengths = np.array(model.link_lengths, dtype=np.float64)
        self.joint_origins = np.array(model.joint_origins, dtype=np.float64)

        # Total reach of the robot
        self.total_reach = np.sum(self.link_lengths)

    def _forward_kinematics(self, thetas: np.ndarray) -> np.ndarray:
        """Compute joint positions from joint angles.

        Args:
            thetas: Joint angles (n_joints,)

        Returns:
            Joint positions as (n_joints + 1, 2) array, including base at origin.
        """
        positions = np.zeros((self.n_joints + 1, 2), dtype=np.float64)
        cumulative_angle = 0.0

        for i in range(self.n_joints):
            cumulative_angle += thetas[i] + self.joint_origins[i]
            positions[i + 1, 0] = positions[i, 0] + self.link_lengths[i] * np.cos(
                cumulative_angle
            )
            positions[i + 1, 1] = positions[i, 1] + self.link_lengths[i] * np.sin(
                cumulative_angle
            )

        return positions

    def _positions_to_angles(self, positions: np.ndarray) -> np.ndarray:
        """Convert joint positions back to joint angles.

        Args:
            positions: Joint positions as (n_joints + 1, 2) array.

        Returns:
            Joint angles as (n_joints,) array.
        """
        angles = np.zeros(self.n_joints, dtype=np.float64)
        cumulative_angle = 0.0

        for i in range(self.n_joints):
            # Vector from joint i to joint i+1
            dx = positions[i + 1, 0] - positions[i, 0]
            dy = positions[i + 1, 1] - positions[i, 1]

            # Absolute angle of this link
            link_angle = np.arctan2(dy, dx)

            # Joint angle is relative to previous cumulative angle and joint origin
            angles[i] = link_angle - cumulative_angle - self.joint_origins[i]

            # Update cumulative angle for next joint
            cumulative_angle = link_angle

        return angles

    def _fabrik_backward(self, positions: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Backward reaching phase: move end effector to target, adjust backward.

        Args:
            positions: Current joint positions (n_joints + 1, 2).
            target: Target position (2,).

        Returns:
            Updated joint positions.
        """
        positions = positions.copy()
        n = len(positions)

        # Set end effector to target
        positions[n - 1] = target

        # Work backward from end effector to base
        for i in range(n - 2, -1, -1):
            # Direction from joint i+1 to joint i
            direction = positions[i] - positions[i + 1]
            dist = np.linalg.norm(direction)

            if dist > 1e-10:
                direction = direction / dist
            else:
                # If points coincide, use previous direction or default
                direction = np.array([1.0, 0.0])

            # Place joint i at link_length distance from joint i+1
            positions[i] = positions[i + 1] + direction * self.link_lengths[i]

        return positions

    def _fabrik_forward(self, positions: np.ndarray, base: np.ndarray) -> np.ndarray:
        """Forward reaching phase: fix base, adjust forward to end effector.

        Args:
            positions: Current joint positions (n_joints + 1, 2).
            base: Base position (2,), typically origin.

        Returns:
            Updated joint positions.
        """
        positions = positions.copy()
        n = len(positions)

        # Fix base position
        positions[0] = base

        # Work forward from base to end effector
        for i in range(n - 1):
            # Direction from joint i to joint i+1
            direction = positions[i + 1] - positions[i]
            dist = np.linalg.norm(direction)

            if dist > 1e-10:
                direction = direction / dist
            else:
                # If points coincide, use a default direction
                direction = np.array([1.0, 0.0])

            # Place joint i+1 at link_length distance from joint i
            positions[i + 1] = positions[i] + direction * self.link_lengths[i]

        return positions

    def _apply_angle_constraint(
        self, positions: np.ndarray, target_angle: float
    ) -> np.ndarray:
        """Apply end effector angle constraint by adjusting the last link.

        Args:
            positions: Joint positions (n_joints + 1, 2).
            target_angle: Desired end effector angle in radians.

        Returns:
            Updated joint positions with angle constraint applied.
        """
        positions = positions.copy()
        n = len(positions)

        # The last link should point in the target_angle direction
        # Adjust the second-to-last joint position to achieve this
        last_link_length = self.link_lengths[-1]

        # Direction the last link should point
        direction = np.array([np.cos(target_angle), np.sin(target_angle)])

        # Move the second-to-last joint so that the last link has the correct angle
        positions[n - 2] = positions[n - 1] - direction * last_link_length

        return positions

    def __call__(
        self,
        state: RobotState,
        desired: DesiredPosition,
        profile: bool = False,
    ) -> RobotState | Tuple[RobotState, IKFabrikProfile]:
        """Solve IK for the desired position using FABRIK.

        Args:
            state: Current robot state.
            desired: Desired end effector position and optional angle.
            profile: If True, return profiling information along with the result.

        Returns:
            If profile is False: RobotState with the solution.
            If profile is True: Tuple of (RobotState, IKFabrikProfile).
        """
        if state.model != self.model:
            raise ValueError("State model does not match IKFabrik model")

        if desired.ee_position is None:
            raise ValueError("DesiredPosition must have an ee_position")

        target = np.array(desired.ee_position, dtype=np.float64)
        base = np.array([0.0, 0.0], dtype=np.float64)

        # Check if target is reachable
        target_distance = np.linalg.norm(target - base)
        if target_distance > self.total_reach:
            # Target is unreachable - stretch toward it
            pass  # FABRIK will naturally stretch toward unreachable targets

        # Initialize joint positions from current state
        current_angles = np.array(state.current.joint_angles, dtype=np.float64)
        positions = self._forward_kinematics(current_angles)

        # Start profiling
        start_time = time.perf_counter()

        initial_error = np.linalg.norm(positions[-1] - target)
        converged = False
        iterations_completed = 0

        for iteration in range(self.max_iterations):
            # Backward reaching: move end effector to target
            positions = self._fabrik_backward(positions, target)

            # Apply angle constraint if specified (during backward pass)
            if desired.ee_angle is not None:
                positions = self._apply_angle_constraint(positions, desired.ee_angle)

            # Forward reaching: fix base position
            positions = self._fabrik_forward(positions, base)

            iterations_completed = iteration + 1

            # Check convergence
            current_error = np.linalg.norm(positions[-1] - target)
            if current_error < self.tolerance:
                converged = True
                break

        # End profiling
        end_time = time.perf_counter()

        # Convert positions back to joint angles
        joint_angles = self._positions_to_angles(positions)

        # Normalize angles to [-pi, pi]
        joint_angles = np.arctan2(np.sin(joint_angles), np.cos(joint_angles))

        result_state = state.with_position(
            RobotPosition(joint_angles=tuple(float(a) for a in joint_angles)),
            desired=desired,
        )

        if profile:
            final_error = float(np.linalg.norm(positions[-1] - target))

            profile_result = IKFabrikProfile(
                solve_time_ms=(end_time - start_time) * 1000,
                iterations=iterations_completed,
                converged=converged,
                initial_error=initial_error,
                final_error=final_error,
                position_error=final_error,
            )
            return result_state, profile_result

        return result_state


if __name__ == "__main__":
    # Interactive IK solver demo using RobotVisualizer
    from visualization import RobotVisualizer

    # Create a 3-link robot
    model = RobotModel(link_lengths=(1.0, 0.8, 0.6))
    world = WorldModel()

    # Create the IK solver
    ik_solver = IKFabrik(model, world=world, max_iterations=100)

    # Initial position
    initial_position = RobotPosition(joint_angles=(0.0, math.pi / 4, -math.pi / 4))
    current_state = RobotState(model, current=initial_position, world=world)

    # Create visualizer
    viz = RobotVisualizer(current_state)

    # Click callback that updates the target and solves IK
    def on_click(x: float, y: float, btn: Literal["left", "right"]):
        global current_state
        print(f"\nClicked at: ({x:.2f}, {y:.2f}) {btn}")

        new_ee_angle: Optional[float] = (
            current_state.desired.ee_angle if current_state.desired else None
        )
        if btn == "right":
            new_ee_angle = 0.0 if new_ee_angle is None else None

        # Solve IK with profiling
        try:
            result = ik_solver(
                current_state,
                DesiredPosition(ee_position=(x, y), ee_angle=new_ee_angle),
                profile=True,
            )
            assert isinstance(result, tuple)
            solution_state, profile = result
            solution = solution_state.current
            print(f"Solution: {tuple(f'{a:.3f}' for a in solution.joint_angles)}")

            # Print profiling information
            print(f"Solve time: {profile.solve_time_ms:.2f}ms")
            print(f"Iterations: {profile.iterations} (converged: {profile.converged})")
            print(f"Error: {profile.initial_error:.6f} -> {profile.final_error:.6f}")
            print(f"Position error: {profile.position_error:.6f}")

            # Update the visualization with the new solution
            current_state = solution_state
            viz.update(current_state)

        except Exception as e:
            print(f"Error solving IK: {e}")
            import traceback

            traceback.print_exc()

    # Set the click callback
    viz.set_click_callback(on_click)

    print("Interactive IK Solver (FABRIK)")
    print("=" * 60)
    print("Click anywhere in the window to set a target position.")
    print("The robot will solve IK and move to reach that target.")
    print("Right-click to toggle angle constraint (0 radians).")
    print("=" * 60)

    # Show the visualization
    viz.show()
