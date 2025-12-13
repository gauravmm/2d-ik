import math
import pytest
from datamodel import RobotModel, RobotPosition


class TestGetJointPositions:
    """Test cases for RobotPosition.get_joint_positions method."""

    def test_single_link_zero_angle(self):
        """Test a single link at zero angle points along positive x-axis."""
        model = RobotModel(link_lengths=(1.0,))
        position = RobotPosition(joint_angles=(0.0,))

        joints = model.forward_kinematics(position)

        assert len(joints) == 2  # Base + end effector
        assert joints[0] == (0.0, 0.0)  # Base at origin
        assert joints[1] == pytest.approx((1.0, 0.0))  # End effector at (1, 0)

    def test_single_link_90_degrees(self):
        """Test a single link at 90 degrees points along positive y-axis."""
        model = RobotModel(link_lengths=(1.0,))
        position = RobotPosition(joint_angles=(math.pi / 2,))

        joints = model.forward_kinematics(position)

        assert len(joints) == 2
        assert joints[0] == (0.0, 0.0)
        assert joints[1] == pytest.approx((0.0, 1.0))

    def test_two_links_straight(self):
        """Test two links in a straight line along x-axis."""
        model = RobotModel(link_lengths=(1.0, 1.0))
        position = RobotPosition(joint_angles=(0.0, 0.0))

        joints = model.forward_kinematics(position)

        assert len(joints) == 3  # Base + 2 joints
        assert joints[0] == (0.0, 0.0)
        assert joints[1] == pytest.approx((1.0, 0.0))
        assert joints[2] == pytest.approx((2.0, 0.0))

    def test_two_links_right_angle(self):
        """Test two links forming a right angle (L-shape)."""
        model = RobotModel(link_lengths=(1.0, 1.0))
        position = RobotPosition(joint_angles=(0.0, math.pi / 2))

        joints = model.forward_kinematics(position)

        assert len(joints) == 3
        assert joints[0] == (0.0, 0.0)
        assert joints[1] == pytest.approx((1.0, 0.0))
        assert joints[2] == pytest.approx((1.0, 1.0))  # Second link points up

    def test_three_links_complex(self):
        """Test three links with various angles."""
        model = RobotModel(link_lengths=(1.0, 1.0, 1.0))
        # First link at 0°, second at +45°, third at -90°
        position = RobotPosition(
            joint_angles=(0.0, math.pi / 4, -math.pi / 2)
        )

        joints = model.forward_kinematics(position)

        assert len(joints) == 4
        assert joints[0] == (0.0, 0.0)
        assert joints[1] == pytest.approx((1.0, 0.0))

        # Second joint: cumulative angle = 0 + π/4
        expected_x2 = 1.0 + math.cos(math.pi / 4)
        expected_y2 = 0.0 + math.sin(math.pi / 4)
        assert joints[2] == pytest.approx((expected_x2, expected_y2))

        # Third joint: cumulative angle = 0 + π/4 - π/2 = -π/4
        expected_x3 = expected_x2 + math.cos(-math.pi / 4)
        expected_y3 = expected_y2 + math.sin(-math.pi / 4)
        assert joints[3] == pytest.approx((expected_x3, expected_y3))

    def test_with_joint_origins(self):
        """Test that joint origins (offsets) are applied correctly."""
        model = RobotModel(
            link_lengths=(1.0, 1.0),
            joint_origins=(math.pi / 4, 0.0)  # First joint has 45° offset
        )
        position = RobotPosition(joint_angles=(0.0, 0.0))

        joints = model.forward_kinematics(position)

        assert len(joints) == 3
        assert joints[0] == (0.0, 0.0)

        # First link with 45° offset
        assert joints[1] == pytest.approx((math.cos(math.pi / 4), math.sin(math.pi / 4)))

        # Second link continues at same angle
        expected_x = 2 * math.cos(math.pi / 4)
        expected_y = 2 * math.sin(math.pi / 4)
        assert joints[2] == pytest.approx((expected_x, expected_y))

    def test_different_link_lengths(self):
        """Test robot with different link lengths."""
        model = RobotModel(link_lengths=(2.0, 1.5, 0.5))
        position = RobotPosition(joint_angles=(0.0, 0.0, 0.0))

        joints = model.forward_kinematics(position)

        assert len(joints) == 4
        assert joints[0] == (0.0, 0.0)
        assert joints[1] == pytest.approx((2.0, 0.0))
        assert joints[2] == pytest.approx((3.5, 0.0))
        assert joints[3] == pytest.approx((4.0, 0.0))

    def test_full_rotation(self):
        """Test that a 360° rotation returns to the same relative position."""
        model = RobotModel(link_lengths=(1.0, 1.0))
        position = RobotPosition(joint_angles=(0.0, 2 * math.pi))

        joints = model.forward_kinematics(position)

        # Second link should complete a full rotation and point along x-axis
        assert joints[2] == pytest.approx((2.0, 0.0))

    def test_negative_angles(self):
        """Test that negative angles work correctly (clockwise rotation)."""
        model = RobotModel(link_lengths=(1.0, 1.0))
        position = RobotPosition(joint_angles=(0.0, -math.pi / 2))

        joints = model.forward_kinematics(position)

        assert len(joints) == 3
        assert joints[0] == (0.0, 0.0)
        assert joints[1] == pytest.approx((1.0, 0.0))
        assert joints[2] == pytest.approx((1.0, -1.0))  # Second link points down

    def test_combined_origins_and_angles(self):
        """Test complex case with both joint origins and angles."""
        model = RobotModel(
            link_lengths=(1.0, 1.0),
            joint_origins=(math.pi / 6, -math.pi / 6)  # 30° and -30° offsets
        )
        position = RobotPosition(
            joint_angles=(math.pi / 6, math.pi / 3)  # 30° and 60° angles
        )

        joints = model.forward_kinematics(position)

        assert len(joints) == 3
        assert joints[0] == (0.0, 0.0)

        # First link: angle = π/6 + π/6 = π/3
        assert joints[1] == pytest.approx((math.cos(math.pi / 3), math.sin(math.pi / 3)))

        # Second link: cumulative = π/3 + π/3 - π/6 = π/2
        expected_x = math.cos(math.pi / 3) + math.cos(math.pi / 2)
        expected_y = math.sin(math.pi / 3) + math.sin(math.pi / 2)
        assert joints[2] == pytest.approx((expected_x, expected_y))
