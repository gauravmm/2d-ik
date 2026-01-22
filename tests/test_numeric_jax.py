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

# Common robot models
TWO_LINK = RobotModel(link_lengths=(1.0, 1.0))
THREE_LINK = RobotModel(link_lengths=(1.0, 0.8, 0.6))
THREE_LINK_LIMITED = RobotModel(
    link_lengths=(1.0, 0.8, 0.6),
    joint_limits=((0.0, math.pi), (-math.pi, 0), (-math.pi / 2, math.pi / 2)),
)


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
    """Solve IK and verify position/angle errors are within tolerance."""
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

    @pytest.mark.parametrize(
        "model,target,initial_angles",
        [
            (TWO_LINK, (1.5, 0.5), (0.5, 0.3)),
            (THREE_LINK, (1.8, 0.8), (0.5, 0.3, -0.2)),
        ],
    )
    def test_basic_ik(self, model, target, initial_angles):
        """Test basic IK on different robot configurations."""
        solve_and_check(model, target=target, initial_angles=initial_angles)

    def test_profiling_output(self):
        """Test that profiling returns expected fields."""
        ik_solver = IKNumericJAX(TWO_LINK, world=None)
        state = RobotState(TWO_LINK, RobotPosition(joint_angles=(0.5, 0.3)))
        result = ik_solver(state, DesiredPosition(ee_position=(1.5, 0.5)), profile=True)
        assert isinstance(result, tuple)
        _, profile = result

        assert profile.solve_time_ms >= 0
        assert profile.iterations > 0
        assert profile.final_loss <= profile.initial_loss


class TestIKNumericJAXAngleConstraints:
    """Test angle constraint functionality."""

    def test_position_and_angle_constraint(self):
        """Test IK with both position and angle constraints."""
        solve_and_check(
            THREE_LINK,
            target=(1.0, 0.5),
            initial_angles=(0.5, 0.3, -0.2),
            target_angle=0.0,
            max_error=0.02,
            max_iterations=1000,
        )

    @pytest.mark.parametrize("target_angle", [0.0, math.pi / 2])
    def test_different_angles_same_position(self, target_angle):
        """Test reaching same position with different end effector angles."""
        solve_and_check(
            THREE_LINK,
            target=(1.5, 0.5),
            initial_angles=(0.5, 0.3, -0.2),
            target_angle=target_angle,
            max_error=0.02,
            lr=0.01,
            max_iterations=200,
        )


class TestIKNumericJAXJointLimits:
    """Test joint limit enforcement."""

    def test_joint_limits_respected(self):
        """Test that joint limits are respected."""
        solution = solve_and_check(
            THREE_LINK_LIMITED,
            target=(1.5, 0.5),
            initial_angles=(0.5, -0.3, 0.2),
            max_iterations=1000,
        )

        for i, (angle, limit) in enumerate(
            zip(solution.joint_angles, THREE_LINK_LIMITED.joint_limits)
        ):
            if limit is not None:
                assert angle >= limit[0], (
                    f"Joint {i} below minimum: {angle} < {limit[0]}"
                )
                assert angle <= limit[1], (
                    f"Joint {i} above maximum: {angle} > {limit[1]}"
                )

    def test_no_joint_limits(self):
        """Test that solver works without joint limits."""
        solve_and_check(THREE_LINK, target=(1.5, 0.5), initial_angles=(0.5, 0.3, -0.2))


class TestIKNumericJAXNogoZones:
    """Test nogo zone collision avoidance."""

    @pytest.mark.parametrize(
        "world,target,max_error",
        [
            (WorldModel(nogo=[RegionHalfspace((0, -1), (0, 0))]), (1.5, 0.5), 0.05),
            (WorldModel(nogo=[RegionBall((1.0, 0.5), 0.2)]), (1.8, 0.3), 0.1),
            (WorldModel(nogo=[RegionRectangle(0.8, 1.2, 0.3, 0.7)]), (1.8, 0.3), 0.1),
        ],
    )
    def test_nogo_avoidance(self, world, target, max_error):
        """Test avoidance of different nogo zone types."""
        solve_and_check(
            THREE_LINK,
            target=target,
            initial_angles=(0.5, 0.3, 0.0),
            world=world,
            max_error=max_error,
            max_iterations=1000,
        )


class TestIKNumericJAXEdgeCases:
    """Test edge cases and error handling."""

    def test_model_mismatch_raises_error(self):
        """Test that mismatched models raise an error."""
        ik_solver = IKNumericJAX(TWO_LINK, world=None)
        other_model = RobotModel(link_lengths=(1.0, 0.8))
        state = RobotState(other_model, RobotPosition(joint_angles=(0.0, 0.0)))

        with pytest.raises(ValueError, match="model"):
            ik_solver(state, DesiredPosition(ee_position=(1.0, 1.0)))

    def test_missing_ee_position_raises_error(self):
        """Test that missing ee_position raises an error."""
        ik_solver = IKNumericJAX(TWO_LINK, world=None)
        state = RobotState(TWO_LINK, RobotPosition(joint_angles=(0.0, 0.0)))

        with pytest.raises(ValueError, match="ee_position"):
            ik_solver(state, DesiredPosition(ee_position=None))  # pyright: ignore[reportArgumentType]

    def test_unreachable_target(self):
        """Test behavior with unreachable target."""
        ik_solver = IKNumericJAX(TWO_LINK, world=None, max_iterations=500)
        state = RobotState(TWO_LINK, RobotPosition(joint_angles=(0.0, 0.0)))

        result = ik_solver(state, DesiredPosition(ee_position=(3.0, 0.0)))
        assert isinstance(result, RobotState)
        end_effector = TWO_LINK.forward_kinematics(result.current)[-1]

        distance_from_origin = math.sqrt(end_effector[0] ** 2 + end_effector[1] ** 2)
        assert abs(distance_from_origin - sum(TWO_LINK.link_lengths)) < 0.1

    def test_line_collision_raises_error(self):
        """Test that line collision geometry raises an error."""
        with pytest.raises(ValueError, match="point"):
            IKNumericJAX(TWO_LINK, world=None, collision_geometry="line")


class TestIKNumericJAXJIT:
    """Test JIT compilation functionality."""

    def test_jit_warmup(self):
        """Test that JIT warmup works."""
        ik_solver = IKNumericJAX(THREE_LINK, world=None, max_iterations=100)
        state = RobotState(THREE_LINK, RobotPosition(joint_angles=(0.5, 0.3, -0.2)))

        result1 = ik_solver(
            state, DesiredPosition(ee_position=(1.5, 0.5)), profile=True
        )
        result2 = ik_solver(
            state, DesiredPosition(ee_position=(1.5, 0.5)), profile=True
        )

        assert isinstance(result1, tuple) and isinstance(result2, tuple)
        assert result1[1].position_error < 0.1
        assert result2[1].position_error < 0.1

    @pytest.mark.parametrize("target", [(1.5, 0.5), (1.0, 1.0), (0.5, 1.5), (1.8, 0.2)])
    def test_multiple_targets(self, target):
        """Test solver with multiple target positions."""
        solve_and_check(
            THREE_LINK,
            target=target,
            initial_angles=(0.5, 0.3, -0.2),
            max_error=0.05,
            max_iterations=200,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
