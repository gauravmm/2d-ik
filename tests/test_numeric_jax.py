#!/usr/bin/env python3
"""Test cases for JAX-based numeric IK solver."""

import math
from typing import Tuple

import pytest

from datamodel import (
    DesiredPosition,
    RegionBall,
    RegionHalfspace,
    RegionRectangle,
    RobotModel,
    RobotPosition,
    RobotState,
    WorldModel,
)
from numeric_jax import IKNumericJAX


def solve_and_check(
    model: RobotModel,
    target: Tuple[float, float],
    initial_angles: Tuple[float, ...] | None = None,
    target_angle: float | None = None,
    world: WorldModel | None = None,
    max_error: float = 0.01,
    max_angle_error: float = 0.1,
    lr: float = 0.05,
    momentum: float = 0.9,
    max_iterations: int = 500,
) -> RobotPosition:
    """Solve IK and verify position/angle errors are within tolerance.

    Args:
        model: Robot model to use.
        target: Target (x, y) position for end effector.
        initial_angles: Initial joint angles. Defaults to zeros.
        target_angle: Optional target end effector angle.
        world: Optional world model with nogo zones.
        max_error: Maximum allowed position error.
        max_angle_error: Maximum allowed angle error (if target_angle set).
        lr: Learning rate for solver.
        momentum: Momentum for solver.
        max_iterations: Maximum solver iterations.

    Returns:
        The solution RobotPosition.
    """
    ik_solver = IKNumericJAX(
        model,
        world=world,
        lr=lr,
        momentum=momentum,
        max_iterations=max_iterations,
        collision_geometry="point",
    )

    if initial_angles is None:
        initial_angles = tuple(0.0 for _ in model.link_lengths)

    state = RobotState(model, RobotPosition(joint_angles=initial_angles), world=world)
    desired = DesiredPosition(ee_position=target, ee_angle=target_angle)

    result = ik_solver(state, desired)
    assert isinstance(result, RobotState)
    solution = result.current

    # Verify position
    end_effector = model.forward_kinematics(solution)[-1]
    error = math.sqrt(
        (end_effector[0] - target[0]) ** 2 + (end_effector[1] - target[1]) ** 2
    )
    assert error < max_error, f"Position error too large: {error}"

    # Verify angle if specified
    if target_angle is not None:
        actual_angle = sum(solution.joint_angles)
        angle_diff = actual_angle - target_angle
        angle_error = abs(math.atan2(math.sin(angle_diff), math.cos(angle_diff)))
        assert angle_error < max_angle_error, f"Angle error too large: {angle_error}"

    return solution


class TestIKNumericJAXBasic:
    """Basic functionality tests for IKNumericJAX."""

    def test_simple_two_link_ik(self):
        """Test IK on a simple 2-link robot."""
        model = RobotModel(link_lengths=(1.0, 1.0))
        solve_and_check(model, target=(1.5, 0.5), initial_angles=(0.5, 0.3))

    def test_three_link_ik(self):
        """Test IK on a 3-link robot."""
        model = RobotModel(link_lengths=(1.0, 0.8, 0.6))
        solve_and_check(model, target=(1.8, 0.8), initial_angles=(0.5, 0.3, -0.2))

    def test_profiling_output(self):
        """Test that profiling returns expected fields."""
        model = RobotModel(link_lengths=(1.0, 1.0))
        ik_solver = IKNumericJAX(model, world=None)

        state = RobotState(model, RobotPosition(joint_angles=(0.5, 0.3)))
        result = ik_solver(state, DesiredPosition(ee_position=(1.5, 0.5)), profile=True)
        assert isinstance(result, tuple)
        _, profile = result

        assert hasattr(profile, "solve_time_ms")
        assert hasattr(profile, "iterations")
        assert hasattr(profile, "converged")
        assert hasattr(profile, "initial_loss")
        assert hasattr(profile, "final_loss")
        assert hasattr(profile, "position_error")

        assert profile.solve_time_ms >= 0
        assert profile.iterations > 0
        assert profile.final_loss <= profile.initial_loss


class TestIKNumericJAXAngleConstraints:
    """Test angle constraint functionality."""

    def test_position_and_angle_constraint(self):
        """Test IK with both position and angle constraints."""
        model = RobotModel(link_lengths=(1.0, 0.8, 0.6))
        solve_and_check(
            model,
            target=(1.0, 0.5),
            initial_angles=(0.5, 0.3, -0.2),
            target_angle=0.0,
            max_error=0.02,
            max_iterations=1000,
        )

    def test_different_angles_same_position(self):
        """Test reaching same position with different end effector angles."""
        model = RobotModel(link_lengths=(1.0, 0.8, 0.6))
        target = (1.5, 0.5)
        initial = (0.5, 0.3, -0.2)

        solution1 = solve_and_check(
            model, target, initial, target_angle=0.0, max_error=0.02, lr=0.01, max_iterations=200
        )
        solution2 = solve_and_check(
            model, target, initial, target_angle=math.pi / 2, max_error=0.02, lr=0.01, max_iterations=200
        )

        # Should have different angles
        angle1 = math.atan2(math.sin(sum(solution1.joint_angles)), math.cos(sum(solution1.joint_angles)))
        angle2 = math.atan2(math.sin(sum(solution2.joint_angles)), math.cos(sum(solution2.joint_angles)))
        assert abs(angle1 - angle2) > 0.5


class TestIKNumericJAXJointLimits:
    """Test joint limit enforcement."""

    def test_joint_limits_respected(self):
        """Test that joint limits are respected."""
        model = RobotModel(
            link_lengths=(1.0, 0.8, 0.6),
            joint_limits=(
                (0.0, math.pi),
                (-math.pi, 0),
                (-math.pi / 2, math.pi / 2),
            ),
        )
        solution = solve_and_check(
            model,
            target=(1.5, 0.5),
            initial_angles=(0.5, -0.3, 0.2),
            max_iterations=1000,
        )

        # Check joint limits are respected (clamping-based enforcement)
        for i, (angle, limit) in enumerate(
            zip(solution.joint_angles, model.joint_limits)
        ):
            if limit is not None:
                assert angle >= limit[0], f"Joint {i} below minimum: {angle} < {limit[0]}"
                assert angle <= limit[1], f"Joint {i} above maximum: {angle} > {limit[1]}"

    def test_no_joint_limits(self):
        """Test that solver works without joint limits."""
        model = RobotModel(link_lengths=(1.0, 0.8, 0.6))
        solve_and_check(model, target=(1.5, 0.5), initial_angles=(0.5, 0.3, -0.2))


class TestIKNumericJAXNogoZones:
    """Test nogo zone collision avoidance."""

    def test_halfspace_nogo(self):
        """Test avoidance of halfspace nogo zone."""
        model = RobotModel(link_lengths=(1.0, 0.8, 0.6))
        world = WorldModel(nogo=[RegionHalfspace((0, -1), (0, 0))])
        solve_and_check(
            model,
            target=(1.5, 0.5),
            initial_angles=(0.5, 0.3, 0.0),
            world=world,
            max_error=0.05,
            max_iterations=1000,
        )

    def test_ball_nogo(self):
        """Test avoidance of ball nogo zone."""
        model = RobotModel(link_lengths=(1.0, 0.8, 0.6))
        world = WorldModel(nogo=[RegionBall((1.0, 0.5), 0.2)])
        solve_and_check(
            model,
            target=(1.8, 0.3),
            initial_angles=(0.5, 0.3, 0.0),
            world=world,
            max_error=0.1,
            max_iterations=1000,
        )

    def test_rectangle_nogo(self):
        """Test avoidance of rectangle nogo zone."""
        model = RobotModel(link_lengths=(1.0, 0.8, 0.6))
        world = WorldModel(nogo=[RegionRectangle(0.8, 1.2, 0.3, 0.7)])
        solve_and_check(
            model,
            target=(1.8, 0.3),
            initial_angles=(0.5, 0.3, 0.0),
            world=world,
            max_error=0.1,
            max_iterations=1000,
        )


class TestIKNumericJAXEdgeCases:
    """Test edge cases and error handling."""

    def test_model_mismatch_raises_error(self):
        """Test that mismatched models raise an error."""
        model1 = RobotModel(link_lengths=(1.0, 1.0))
        model2 = RobotModel(link_lengths=(1.0, 0.8))
        ik_solver = IKNumericJAX(model1, world=None)
        state = RobotState(model2, RobotPosition(joint_angles=(0.0, 0.0)))

        with pytest.raises(ValueError, match="model"):
            ik_solver(state, DesiredPosition(ee_position=(1.0, 1.0)))

    def test_missing_ee_position_raises_error(self):
        """Test that missing ee_position raises an error."""
        model = RobotModel(link_lengths=(1.0, 1.0))
        ik_solver = IKNumericJAX(model, world=None)
        state = RobotState(model, RobotPosition(joint_angles=(0.0, 0.0)))

        with pytest.raises(ValueError, match="ee_position"):
            ik_solver(state, DesiredPosition(ee_position=None))

    def test_unreachable_target(self):
        """Test behavior with unreachable target."""
        model = RobotModel(link_lengths=(1.0, 1.0))
        ik_solver = IKNumericJAX(model, world=None, max_iterations=500)
        state = RobotState(model, RobotPosition(joint_angles=(0.0, 0.0)))

        # Target outside max reach of 2.0
        result = ik_solver(state, DesiredPosition(ee_position=(3.0, 0.0)))
        assert isinstance(result, RobotState)
        end_effector = model.forward_kinematics(result.current)[-1]

        # Should extend toward target (approximately at max reach)
        distance_from_origin = math.sqrt(end_effector[0] ** 2 + end_effector[1] ** 2)
        assert abs(distance_from_origin - sum(model.link_lengths)) < 0.1

    def test_line_collision_raises_error(self):
        """Test that line collision geometry raises an error."""
        model = RobotModel(link_lengths=(1.0, 1.0))
        with pytest.raises(ValueError, match="point"):
            IKNumericJAX(model, world=None, collision_geometry="line")


class TestIKNumericJAXJIT:
    """Test JIT compilation functionality."""

    def test_jit_warmup(self):
        """Test that JIT warmup works."""
        model = RobotModel(link_lengths=(1.0, 0.8, 0.6))
        ik_solver = IKNumericJAX(model, world=None, max_iterations=100)
        state = RobotState(model, RobotPosition(joint_angles=(0.5, 0.3, -0.2)))
        target = (1.5, 0.5)

        # First call (includes JIT compilation)
        result1 = ik_solver(state, DesiredPosition(ee_position=target), profile=True)
        assert isinstance(result1, tuple)
        _, profile1 = result1

        # Second call (should use cached JIT)
        result2 = ik_solver(state, DesiredPosition(ee_position=target), profile=True)
        assert isinstance(result2, tuple)
        _, profile2 = result2

        # Both should produce valid results
        assert profile1.position_error < 0.1
        assert profile2.position_error < 0.1

    def test_multiple_calls_same_solver(self):
        """Test multiple calls to the same solver instance."""
        model = RobotModel(link_lengths=(1.0, 0.8, 0.6))
        for target in [(1.5, 0.5), (1.0, 1.0), (0.5, 1.5), (1.8, 0.2)]:
            solve_and_check(
                model,
                target=target,
                initial_angles=(0.5, 0.3, -0.2),
                max_error=0.05,
                max_iterations=200,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
