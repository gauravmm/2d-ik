#!/usr/bin/env python3
"""Test cases for JAX-based numeric IK solver."""

import math
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

# Skip all tests if JAX is not installed
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from numeric_jax import IKNumericJAX, IKNumericJAXProfile


class TestIKNumericJAXBasic:
    """Basic functionality tests for IKNumericJAX."""

    @pytest.fixture
    def two_link_robot(self):
        """Create a simple 2-link robot for testing."""
        return RobotModel(link_lengths=(1.0, 1.0))

    @pytest.fixture
    def three_link_robot(self):
        """Create a standard 3-link robot for testing."""
        return RobotModel(link_lengths=(1.0, 0.8, 0.6))

    def test_simple_two_link_ik(self, two_link_robot):
        """Test IK on a simple 2-link robot."""
        model = two_link_robot
        ik_solver = IKNumericJAX(
            model, world=None, lr=0.05, momentum=0.9, max_iterations=500
        )

        initial_position = RobotPosition(joint_angles=(0.5, 0.3))
        target = (1.5, 0.5)
        state = RobotState(model, initial_position)

        result = ik_solver(state, DesiredPosition(ee_position=target), profile=True)
        assert isinstance(result, tuple)
        solution_state, profile = result
        solution = solution_state.current

        # Verify the solution
        end_effector_positions = model.forward_kinematics(solution)
        end_effector = end_effector_positions[-1]

        error = math.sqrt(
            (end_effector[0] - target[0]) ** 2 + (end_effector[1] - target[1]) ** 2
        )

        assert error < 0.01, f"Position error too large: {error}"
        assert isinstance(profile, IKNumericJAXProfile)

    def test_three_link_ik(self, three_link_robot):
        """Test IK on a 3-link robot."""
        model = three_link_robot
        ik_solver = IKNumericJAX(
            model, world=None, lr=0.05, momentum=0.9, max_iterations=500
        )

        initial_position = RobotPosition(joint_angles=(0.5, 0.3, -0.2))
        target = (1.8, 0.8)
        state = RobotState(model, initial_position)

        solution_state = ik_solver(state, DesiredPosition(ee_position=target))
        solution = solution_state.current

        end_effector_positions = model.forward_kinematics(solution)
        end_effector = end_effector_positions[-1]

        error = math.sqrt(
            (end_effector[0] - target[0]) ** 2 + (end_effector[1] - target[1]) ** 2
        )

        assert error < 0.01, f"Position error too large: {error}"

    def test_profiling_output(self, two_link_robot):
        """Test that profiling returns expected fields."""
        model = two_link_robot
        ik_solver = IKNumericJAX(model, world=None)

        initial_position = RobotPosition(joint_angles=(0.5, 0.3))
        target = (1.5, 0.5)
        state = RobotState(model, initial_position)

        result = ik_solver(state, DesiredPosition(ee_position=target), profile=True)
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

    @pytest.fixture
    def three_link_robot(self):
        """Create a standard 3-link robot for testing."""
        return RobotModel(link_lengths=(1.0, 0.8, 0.6))

    def test_position_and_angle_constraint(self, three_link_robot):
        """Test IK with both position and angle constraints."""
        model = three_link_robot
        ik_solver = IKNumericJAX(
            model, world=None, lr=0.05, momentum=0.9, max_iterations=1000
        )

        target_pos = (1.5, 0.5)
        target_angle = 0.0

        initial_position = RobotPosition(joint_angles=(0.5, 0.3, -0.2))
        state = RobotState(model, initial_position)

        solution_state = ik_solver(
            state, DesiredPosition(ee_position=target_pos, ee_angle=target_angle)
        )
        solution = solution_state.current

        positions = model.forward_kinematics(solution)
        end_effector_pos = positions[-1]

        position_error = math.sqrt(
            (end_effector_pos[0] - target_pos[0]) ** 2
            + (end_effector_pos[1] - target_pos[1]) ** 2
        )

        # Compute actual angle (sum of joint angles for this simple robot)
        actual_angle = sum(solution.joint_angles)
        angle_diff = actual_angle - target_angle
        angle_error = math.atan2(math.sin(angle_diff), math.cos(angle_diff))

        assert position_error < 0.02, f"Position error too large: {position_error}"
        assert abs(angle_error) < 0.1, f"Angle error too large: {angle_error}"

    def test_different_angles_same_position(self, three_link_robot):
        """Test reaching same position with different end effector angles."""
        model = three_link_robot
        ik_solver = IKNumericJAX(
            model, world=None, lr=0.05, momentum=0.9, max_iterations=1000
        )

        target_pos = (1.5, 0.5)
        initial_position = RobotPosition(joint_angles=(0.5, 0.3, -0.2))
        state = RobotState(model, initial_position)

        # Solve for angle 0
        solution1 = ik_solver(
            state, DesiredPosition(ee_position=target_pos, ee_angle=0.0)
        ).current
        angle1 = sum(solution1.joint_angles)

        # Solve for angle pi/2
        solution2 = ik_solver(
            state, DesiredPosition(ee_position=target_pos, ee_angle=math.pi / 2)
        ).current
        angle2 = sum(solution2.joint_angles)

        # Both should reach target
        pos1 = model.forward_kinematics(solution1)[-1]
        pos2 = model.forward_kinematics(solution2)[-1]

        error1 = math.sqrt(
            (pos1[0] - target_pos[0]) ** 2 + (pos1[1] - target_pos[1]) ** 2
        )
        error2 = math.sqrt(
            (pos2[0] - target_pos[0]) ** 2 + (pos2[1] - target_pos[1]) ** 2
        )

        assert error1 < 0.02
        assert error2 < 0.02

        # Should have different angles (approximately)
        angle1_wrapped = math.atan2(math.sin(angle1), math.cos(angle1))
        angle2_wrapped = math.atan2(math.sin(angle2), math.cos(angle2))
        assert abs(angle1_wrapped - angle2_wrapped) > 0.5


class TestIKNumericJAXJointLimits:
    """Test joint limit enforcement."""

    def test_joint_limits_respected(self):
        """Test that joint limits are approximately respected."""
        model = RobotModel(
            link_lengths=(1.0, 0.8, 0.6),
            joint_limits=(
                (0.0, math.pi),
                (-math.pi, 0),
                (-math.pi / 2, math.pi / 2),
            ),
        )
        ik_solver = IKNumericJAX(
            model, world=None, lr=0.05, momentum=0.9, max_iterations=1000
        )

        initial_position = RobotPosition(joint_angles=(0.5, -0.3, 0.2))
        target = (1.5, 0.5)
        state = RobotState(model, initial_position)

        solution_state = ik_solver(state, DesiredPosition(ee_position=target))
        solution = solution_state.current

        # Check joint limits are approximately respected (soft constraints)
        tolerance = 0.1  # Allow small violations due to penalty-based enforcement
        for i, (angle, limit) in enumerate(
            zip(solution.joint_angles, model.joint_limits)
        ):
            if limit is not None:
                assert angle >= limit[0] - tolerance, (
                    f"Joint {i} below minimum: {angle} < {limit[0]}"
                )
                assert angle <= limit[1] + tolerance, (
                    f"Joint {i} above maximum: {angle} > {limit[1]}"
                )

    def test_no_joint_limits(self):
        """Test that solver works without joint limits."""
        model = RobotModel(link_lengths=(1.0, 0.8, 0.6))
        ik_solver = IKNumericJAX(model, world=None)

        initial_position = RobotPosition(joint_angles=(0.0, 0.0, 0.0))
        target = (1.5, 0.5)
        state = RobotState(model, initial_position)

        solution_state = ik_solver(state, DesiredPosition(ee_position=target))
        solution = solution_state.current

        positions = model.forward_kinematics(solution)
        end_effector = positions[-1]

        error = math.sqrt(
            (end_effector[0] - target[0]) ** 2 + (end_effector[1] - target[1]) ** 2
        )
        assert error < 0.01


class TestIKNumericJAXNogoZones:
    """Test nogo zone collision avoidance."""

    @pytest.fixture
    def three_link_robot(self):
        """Create a standard 3-link robot for testing."""
        return RobotModel(link_lengths=(1.0, 0.8, 0.6))

    def test_halfspace_nogo(self, three_link_robot):
        """Test avoidance of halfspace nogo zone."""
        model = three_link_robot
        # Create a halfspace that blocks y < 0
        world = WorldModel(nogo=[RegionHalfspace((0, -1), (0, 0))])

        ik_solver = IKNumericJAX(
            model,
            world=world,
            collision_geometry="point",
            lr=0.05,
            momentum=0.9,
            max_iterations=1000,
        )

        initial_position = RobotPosition(joint_angles=(0.5, 0.3, 0.0))
        # Target in valid region
        target = (1.5, 0.5)
        state = RobotState(model, initial_position, world=world)

        solution_state = ik_solver(state, DesiredPosition(ee_position=target))
        solution = solution_state.current

        # Verify end effector is close to target
        positions = model.forward_kinematics(solution)
        end_effector = positions[-1]

        error = math.sqrt(
            (end_effector[0] - target[0]) ** 2 + (end_effector[1] - target[1]) ** 2
        )
        assert error < 0.05  # May not be exact due to nogo constraints

    def test_ball_nogo(self, three_link_robot):
        """Test avoidance of ball nogo zone."""
        model = three_link_robot
        # Create a ball obstacle at (1.0, 0.5) with radius 0.2
        world = WorldModel(nogo=[RegionBall((1.0, 0.5), 0.2)])

        ik_solver = IKNumericJAX(
            model,
            world=world,
            collision_geometry="point",
            lr=0.05,
            momentum=0.9,
            max_iterations=1000,
        )

        initial_position = RobotPosition(joint_angles=(0.5, 0.3, 0.0))
        target = (1.8, 0.3)
        state = RobotState(model, initial_position, world=world)

        solution_state = ik_solver(state, DesiredPosition(ee_position=target))
        solution = solution_state.current

        positions = model.forward_kinematics(solution)
        end_effector = positions[-1]

        error = math.sqrt(
            (end_effector[0] - target[0]) ** 2 + (end_effector[1] - target[1]) ** 2
        )
        # Check we got reasonably close
        assert error < 0.1

    def test_rectangle_nogo(self, three_link_robot):
        """Test avoidance of rectangle nogo zone."""
        model = three_link_robot
        # Create a rectangle obstacle
        world = WorldModel(nogo=[RegionRectangle(0.8, 1.2, 0.3, 0.7)])

        ik_solver = IKNumericJAX(
            model,
            world=world,
            collision_geometry="point",
            lr=0.05,
            momentum=0.9,
            max_iterations=1000,
        )

        initial_position = RobotPosition(joint_angles=(0.5, 0.3, 0.0))
        target = (1.8, 0.3)
        state = RobotState(model, initial_position, world=world)

        solution_state = ik_solver(state, DesiredPosition(ee_position=target))
        solution = solution_state.current

        positions = model.forward_kinematics(solution)
        end_effector = positions[-1]

        error = math.sqrt(
            (end_effector[0] - target[0]) ** 2 + (end_effector[1] - target[1]) ** 2
        )
        assert error < 0.1


class TestIKNumericJAXEdgeCases:
    """Test edge cases and error handling."""

    def test_model_mismatch_raises_error(self):
        """Test that mismatched models raise an error."""
        model1 = RobotModel(link_lengths=(1.0, 1.0))
        model2 = RobotModel(link_lengths=(1.0, 0.8))

        ik_solver = IKNumericJAX(model1, world=None)

        initial_position = RobotPosition(joint_angles=(0.0, 0.0))
        state = RobotState(model2, initial_position)

        with pytest.raises(ValueError, match="model"):
            ik_solver(state, DesiredPosition(ee_position=(1.0, 1.0)))

    def test_missing_ee_position_raises_error(self):
        """Test that missing ee_position raises an error."""
        model = RobotModel(link_lengths=(1.0, 1.0))
        ik_solver = IKNumericJAX(model, world=None)

        initial_position = RobotPosition(joint_angles=(0.0, 0.0))
        state = RobotState(model, initial_position)

        with pytest.raises(ValueError, match="ee_position"):
            ik_solver(state, DesiredPosition(ee_position=None))

    def test_unreachable_target(self):
        """Test behavior with unreachable target."""
        model = RobotModel(link_lengths=(1.0, 1.0))
        ik_solver = IKNumericJAX(model, world=None, max_iterations=500)

        initial_position = RobotPosition(joint_angles=(0.0, 0.0))
        # Target outside max reach of 2.0
        target = (3.0, 0.0)
        state = RobotState(model, initial_position)

        solution_state = ik_solver(state, DesiredPosition(ee_position=target))
        solution = solution_state.current

        positions = model.forward_kinematics(solution)
        end_effector = positions[-1]

        # Should extend toward target (approximately at max reach)
        distance_from_origin = math.sqrt(end_effector[0] ** 2 + end_effector[1] ** 2)
        max_reach = sum(model.link_lengths)

        # Should be close to max reach
        assert abs(distance_from_origin - max_reach) < 0.1

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

        initial_position = RobotPosition(joint_angles=(0.5, 0.3, -0.2))
        target = (1.5, 0.5)
        state = RobotState(model, initial_position)

        # First call (includes JIT compilation)
        result1 = ik_solver(
            state, DesiredPosition(ee_position=target), profile=True
        )
        assert isinstance(result1, tuple)
        _, profile1 = result1

        # Second call (should use cached JIT)
        result2 = ik_solver(
            state, DesiredPosition(ee_position=target), profile=True
        )
        assert isinstance(result2, tuple)
        _, profile2 = result2

        # Both should produce valid results
        assert profile1.position_error < 0.1
        assert profile2.position_error < 0.1

    def test_multiple_calls_same_solver(self):
        """Test multiple calls to the same solver instance."""
        model = RobotModel(link_lengths=(1.0, 0.8, 0.6))
        ik_solver = IKNumericJAX(model, world=None, max_iterations=200)

        initial_position = RobotPosition(joint_angles=(0.5, 0.3, -0.2))
        state = RobotState(model, initial_position)

        targets = [(1.5, 0.5), (1.0, 1.0), (0.5, 1.5), (1.8, 0.2)]

        for target in targets:
            solution_state = ik_solver(state, DesiredPosition(ee_position=target))
            solution = solution_state.current

            positions = model.forward_kinematics(solution)
            end_effector = positions[-1]

            error = math.sqrt(
                (end_effector[0] - target[0]) ** 2 + (end_effector[1] - target[1]) ** 2
            )
            assert error < 0.05, f"Failed for target {target}: error = {error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
