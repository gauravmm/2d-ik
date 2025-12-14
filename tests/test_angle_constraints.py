#!/usr/bin/env python3
"""Test cases for end_effector_angle constraint functionality."""

import math
import pytest
from datamodel import RobotModel, RobotPosition, RobotState
from symbolic import IKSymbolic


class TestAngleConstraints:
    """Test suite for IK solving with end effector angle constraints."""

    @pytest.fixture
    def three_link_robot(self):
        """Create a standard 3-link robot for testing."""
        return RobotModel(link_lengths=(1.0, 0.8, 0.6))

    @pytest.fixture
    def two_link_robot(self):
        """Create a simple 2-link robot for testing."""
        return RobotModel(link_lengths=(1.0, 1.0))

    def test_position_and_angle_constraint(self, three_link_robot):
        """Test IK solving with both position and angle constraints."""
        model = three_link_robot
        ik_solver = IKSymbolic(model)

        target_pos = (1.5, 0.5)
        target_angle = math.pi / 4  # 45 degrees

        initial_position = RobotPosition(joint_angles=(0.0, 0.0, 0.0))

        state = RobotState(
            model=model,
            current=initial_position,
            desired_end_effector=target_pos,
            desired_end_effector_angle=target_angle,
        )

        solution = ik_solver(state)

        # Verify the solution
        positions = model.forward_kinematics(solution)
        end_effector_pos = positions[-1]

        # Compute the actual end effector angle
        actual_angle = ik_solver.angle_func(*solution.joint_angles)

        # Check position error
        position_error = math.sqrt(
            (end_effector_pos[0] - target_pos[0]) ** 2
            + (end_effector_pos[1] - target_pos[1]) ** 2
        )

        # Compute angle error with wrapping
        angle_diff = actual_angle - target_angle
        angle_error = math.atan2(math.sin(angle_diff), math.cos(angle_diff))

        assert position_error < 0.01, f"Position error too large: {position_error}"
        assert abs(angle_error) < 0.01, f"Angle error too large: {angle_error}"

    def test_angle_wrapping_positive(self, three_link_robot):
        """Test angle wrapping near +π boundary."""
        model = three_link_robot
        ik_solver = IKSymbolic(model)

        target_pos = (0.5, 0.5)
        target_angle = 3.0  # Near π

        initial_position = RobotPosition(joint_angles=(0.5, 0.5, 0.5))

        state = RobotState(
            model=model,
            current=initial_position,
            desired_end_effector=target_pos,
            desired_end_effector_angle=target_angle,
        )

        solution = ik_solver(state)

        positions = model.forward_kinematics(solution)
        end_effector_pos = positions[-1]
        actual_angle = ik_solver.angle_func(*solution.joint_angles)

        position_error = math.sqrt(
            (end_effector_pos[0] - target_pos[0]) ** 2
            + (end_effector_pos[1] - target_pos[1]) ** 2
        )

        angle_diff = actual_angle - target_angle
        angle_error = math.atan2(math.sin(angle_diff), math.cos(angle_diff))

        assert position_error < 0.01
        assert abs(angle_error) < 0.01

    def test_angle_wrapping_negative(self, three_link_robot):
        """Test angle wrapping near -π boundary."""
        model = three_link_robot
        ik_solver = IKSymbolic(model)

        target_pos = (0.5, 0.5)
        target_angle = -2.8  # Near -π

        initial_position = RobotPosition(joint_angles=(0.5, 0.5, 0.5))

        state = RobotState(
            model=model,
            current=initial_position,
            desired_end_effector=target_pos,
            desired_end_effector_angle=target_angle,
        )

        solution = ik_solver(state)

        positions = model.forward_kinematics(solution)
        end_effector_pos = positions[-1]
        actual_angle = ik_solver.angle_func(*solution.joint_angles)

        position_error = math.sqrt(
            (end_effector_pos[0] - target_pos[0]) ** 2
            + (end_effector_pos[1] - target_pos[1]) ** 2
        )

        angle_diff = actual_angle - target_angle
        angle_error = math.atan2(math.sin(angle_diff), math.cos(angle_diff))

        assert position_error < 0.01
        assert abs(angle_error) < 0.01

    def test_angle_wrapping_across_boundary(self, three_link_robot):
        """Test angle wrapping when target and solution are on opposite sides of ±π."""
        model = three_link_robot
        ik_solver = IKSymbolic(model)

        target_pos = (0.8, 0.3)
        # Target angle close to -π
        target_angle = -math.pi + 0.1

        initial_position = RobotPosition(joint_angles=(0.0, 0.0, 0.0))

        state = RobotState(
            model=model,
            current=initial_position,
            desired_end_effector=target_pos,
            desired_end_effector_angle=target_angle,
        )

        solution = ik_solver(state)
        actual_angle = ik_solver.angle_func(*solution.joint_angles)

        # The angle error should be small even if actual_angle is close to +π
        angle_diff = actual_angle - target_angle
        angle_error = math.atan2(math.sin(angle_diff), math.cos(angle_diff))

        assert abs(angle_error) < 0.01

    def test_position_only_backward_compatibility(self, three_link_robot):
        """Test that position-only mode works when desired_end_effector_angle is None."""
        model = three_link_robot
        ik_solver = IKSymbolic(model)

        target_pos = (1.0, 1.0)

        initial_position = RobotPosition(joint_angles=(0.0, 0.0, 0.0))

        state = RobotState(
            model=model,
            current=initial_position,
            desired_end_effector=target_pos,
            desired_end_effector_angle=None,  # No angle constraint
        )

        solution = ik_solver(state)

        positions = model.forward_kinematics(solution)
        end_effector_pos = positions[-1]

        position_error = math.sqrt(
            (end_effector_pos[0] - target_pos[0]) ** 2
            + (end_effector_pos[1] - target_pos[1]) ** 2
        )

        assert position_error < 0.01

    def test_same_position_different_angles(self, three_link_robot):
        """Test that the same position can be reached with different end effector angles."""
        model = three_link_robot
        ik_solver = IKSymbolic(model)

        target_pos = (1.5, 0.5)
        initial_position = RobotPosition(joint_angles=(0.0, 0.0, 0.0))

        # Solve for 45 degrees
        state1 = RobotState(
            model=model,
            current=initial_position,
            desired_end_effector=target_pos,
            desired_end_effector_angle=math.pi / 4,
        )
        solution1 = ik_solver(state1)
        angle1 = ik_solver.angle_func(*solution1.joint_angles)

        # Solve for 90 degrees
        state2 = RobotState(
            model=model,
            current=initial_position,
            desired_end_effector=target_pos,
            desired_end_effector_angle=math.pi / 2,
        )
        solution2 = ik_solver(state2)
        angle2 = ik_solver.angle_func(*solution2.joint_angles)

        # Both should reach the target position
        pos1 = model.forward_kinematics(solution1)[-1]
        pos2 = model.forward_kinematics(solution2)[-1]

        error1 = math.sqrt(
            (pos1[0] - target_pos[0]) ** 2 + (pos1[1] - target_pos[1]) ** 2
        )
        error2 = math.sqrt(
            (pos2[0] - target_pos[0]) ** 2 + (pos2[1] - target_pos[1]) ** 2
        )

        assert error1 < 0.01
        assert error2 < 0.01

        # But should have different angles
        assert abs(angle1 - math.pi / 4) < 0.01
        assert abs(angle2 - math.pi / 2) < 0.01
        assert abs(angle1 - angle2) > 0.1  # Significantly different

    def test_two_link_robot_angle_constraint(self, two_link_robot):
        """Test angle constraint on a simpler 2-link robot.

        Note: With only 2 links, many position+angle combinations are impossible
        to satisfy exactly. This test verifies the solver finds a best compromise.
        """
        model: RobotModel = two_link_robot
        ik_solver = IKSymbolic(model)

        # For a 2-link robot with links of length 1.0 each,
        # choose a position and angle that can be satisfied together
        # At angle 0, the robot can reach (0, 2.0) when fully extended
        target_pos = (model.link_lengths[1], model.link_lengths[0])
        target_angle = 0.0  # Straight out (horizontal)

        initial_position = RobotPosition(joint_angles=(0.0, 0.0))

        state = RobotState(
            model=model,
            current=initial_position,
            desired_end_effector=target_pos,
            desired_end_effector_angle=target_angle,
        )

        solution = ik_solver(state)

        positions = model.forward_kinematics(solution)
        end_effector_pos = positions[-1]
        actual_angle = ik_solver.angle_func(*solution.joint_angles)

        position_error = math.sqrt(
            (end_effector_pos[0] - target_pos[0]) ** 2
            + (end_effector_pos[1] - target_pos[1]) ** 2
        )

        angle_diff = actual_angle - target_angle
        angle_error = math.atan2(math.sin(angle_diff), math.cos(angle_diff))

        # With 2 links, we can satisfy both constraints when they're compatible
        assert position_error < 0.01
        assert abs(angle_error) < 0.01

    def test_zero_angle_constraint(self, three_link_robot):
        """Test that zero angle (pointing right) works correctly."""
        model = three_link_robot
        ik_solver = IKSymbolic(model)

        target_pos = (2.0, 0.2)
        target_angle = 0.0  # Pointing right

        initial_position = RobotPosition(joint_angles=(0.0, 0.0, 0.0))

        state = RobotState(
            model=model,
            current=initial_position,
            desired_end_effector=target_pos,
            desired_end_effector_angle=target_angle,
        )

        solution = ik_solver(state)

        positions = model.forward_kinematics(solution)
        end_effector_pos = positions[-1]
        actual_angle = ik_solver.angle_func(*solution.joint_angles)

        position_error = math.sqrt(
            (end_effector_pos[0] - target_pos[0]) ** 2
            + (end_effector_pos[1] - target_pos[1]) ** 2
        )

        angle_error = abs(actual_angle - target_angle)

        assert position_error < 0.01
        assert angle_error < 0.01

    def test_pi_angle_constraint(self, three_link_robot):
        """Test that π angle (pointing left) works correctly."""
        model = three_link_robot
        ik_solver = IKSymbolic(model)

        target_pos = (-1.0, 0.5)
        target_angle = math.pi  # Pointing left

        initial_position = RobotPosition(joint_angles=(0.0, 0.0, 0.0))

        state = RobotState(
            model=model,
            current=initial_position,
            desired_end_effector=target_pos,
            desired_end_effector_angle=target_angle,
        )

        solution = ik_solver(state)

        positions = model.forward_kinematics(solution)
        end_effector_pos = positions[-1]
        actual_angle = ik_solver.angle_func(*solution.joint_angles)

        position_error = math.sqrt(
            (end_effector_pos[0] - target_pos[0]) ** 2
            + (end_effector_pos[1] - target_pos[1]) ** 2
        )

        # Handle wrapping for π angle
        angle_diff = actual_angle - target_angle
        angle_error = math.atan2(math.sin(angle_diff), math.cos(angle_diff))

        assert position_error < 0.01
        assert abs(angle_error) < 0.01

    def test_different_initial_positions(self, three_link_robot):
        """Test that angle constraint works from different initial configurations."""
        model = three_link_robot
        ik_solver = IKSymbolic(model)

        target_pos = (1.2, 0.6)
        target_angle = math.pi / 6  # 30 degrees

        # Test from multiple initial positions
        initial_configs = [
            (0.0, 0.0, 0.0),
            (0.5, -0.5, 0.5),
            (1.0, 1.0, -1.0),
            (-0.3, 0.8, -0.5),
        ]

        for initial_angles in initial_configs:
            initial_position = RobotPosition(joint_angles=initial_angles)

            state = RobotState(
                model=model,
                current=initial_position,
                desired_end_effector=target_pos,
                desired_end_effector_angle=target_angle,
            )

            solution = ik_solver(state)

            positions = model.forward_kinematics(solution)
            end_effector_pos = positions[-1]
            actual_angle = ik_solver.angle_func(*solution.joint_angles)

            position_error = math.sqrt(
                (end_effector_pos[0] - target_pos[0]) ** 2
                + (end_effector_pos[1] - target_pos[1]) ** 2
            )

            angle_diff = actual_angle - target_angle
            angle_error = math.atan2(math.sin(angle_diff), math.cos(angle_diff))

            assert position_error < 0.01, f"Failed from initial: {initial_angles}"
            assert abs(angle_error) < 0.01, f"Failed from initial: {initial_angles}"

    def test_robot_with_joint_origins(self):
        """Test angle constraint with a robot that has non-zero joint origins."""
        # Create a robot with joint origin offsets
        model = RobotModel(
            link_lengths=(1.0, 0.8, 0.6), joint_origins=(0.1, -0.2, 0.15)
        )
        ik_solver = IKSymbolic(model)

        target_pos = (1.0, 0.5)
        target_angle = math.pi / 4

        initial_position = RobotPosition(joint_angles=(0.0, 0.0, 0.0))

        state = RobotState(
            model=model,
            current=initial_position,
            desired_end_effector=target_pos,
            desired_end_effector_angle=target_angle,
        )

        solution = ik_solver(state)

        positions = model.forward_kinematics(solution)
        end_effector_pos = positions[-1]
        actual_angle = ik_solver.angle_func(*solution.joint_angles)

        position_error = math.sqrt(
            (end_effector_pos[0] - target_pos[0]) ** 2
            + (end_effector_pos[1] - target_pos[1]) ** 2
        )

        angle_diff = actual_angle - target_angle
        angle_error = math.atan2(math.sin(angle_diff), math.cos(angle_diff))

        assert position_error < 0.01
        assert abs(angle_error) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
