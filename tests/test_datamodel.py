# pyright: reportArgumentType=false

import math

import pytest
import sympy as sp

from datamodel import RegionBall, RegionHalfspace, RobotModel, RobotPosition


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
        position = RobotPosition(joint_angles=(0.0, math.pi / 4, -math.pi / 2))

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
            joint_origins=(math.pi / 4, 0.0),  # First joint has 45° offset
        )
        position = RobotPosition(joint_angles=(0.0, 0.0))

        joints = model.forward_kinematics(position)

        assert len(joints) == 3
        assert joints[0] == (0.0, 0.0)

        # First link with 45° offset
        assert joints[1] == pytest.approx(
            (math.cos(math.pi / 4), math.sin(math.pi / 4))
        )

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
            joint_origins=(math.pi / 6, -math.pi / 6),  # 30° and -30° offsets
        )
        position = RobotPosition(
            joint_angles=(math.pi / 6, math.pi / 3)  # 30° and 60° angles
        )

        joints = model.forward_kinematics(position)

        assert len(joints) == 3
        assert joints[0] == (0.0, 0.0)

        # First link: angle = π/6 + π/6 = π/3
        assert joints[1] == pytest.approx(
            (math.cos(math.pi / 3), math.sin(math.pi / 3))
        )

        # Second link: cumulative = π/3 + π/3 - π/6 = π/2
        expected_x = math.cos(math.pi / 3) + math.cos(math.pi / 2)
        expected_y = math.sin(math.pi / 3) + math.sin(math.pi / 2)
        assert joints[2] == pytest.approx((expected_x, expected_y))


class TestRegionHalfspace:
    """Test cases for RegionHalfspace.point method."""

    def test_point_on_boundary(self):
        """Test that a point on the boundary returns zero residual."""
        # Halfspace with normal pointing right (1, 0), anchor at origin
        region = RegionHalfspace(normal=(1.0, 0.0), anchor=(0.0, 0.0))

        # Point on the boundary (at anchor)
        result = region.point((0.0, 0.0))
        assert result == 0.0

    def test_point_inside_halfspace(self):
        """Test that points inside the halfspace return positive residual."""
        # Halfspace with normal pointing right (1, 0), anchor at origin
        region = RegionHalfspace(normal=(1.0, 0.0), anchor=(0.0, 0.0))

        # Point to the right (inside)
        result = region.point((2.0, 0.0))
        assert result == 2.0

        # Another point inside
        result = region.point((5.0, 3.0))
        assert result == 5.0  # Only x-component matters with normal (1,0)

    def test_point_outside_halfspace(self):
        """Test that points outside the halfspace return negative residual."""
        # Halfspace with normal pointing right (1, 0), anchor at origin
        region = RegionHalfspace(normal=(1.0, 0.0), anchor=(0.0, 0.0))

        # Point to the left (outside)
        result = region.point((-3.0, 0.0))
        assert result == -3.0

        # Another point outside
        result = region.point((-1.0, 5.0))
        assert result == -1.0

    def test_vertical_halfspace(self):
        """Test halfspace with vertical boundary."""
        # Normal pointing up (0, 1), anchor at (2, 1)
        region = RegionHalfspace(normal=(0.0, 1.0), anchor=(2.0, 1.0))

        # Point above boundary (inside)
        result = region.point((2.0, 4.0))
        assert result == 3.0

        # Point below boundary (outside)
        result = region.point((2.0, 0.0))
        assert result == -1.0

        # Point on boundary
        result = region.point((5.0, 1.0))  # x doesn't matter
        assert result == 0.0

    def test_diagonal_halfspace(self):
        """Test halfspace with diagonal boundary."""
        # Normal at 45 degrees: (1/√2, 1/√2)
        normal = (1.0 / math.sqrt(2), 1.0 / math.sqrt(2))
        region = RegionHalfspace(normal=normal, anchor=(0.0, 0.0))

        # Point in direction of normal (inside)
        result = region.point((1.0, 1.0))
        expected = normal[0] * 1.0 + normal[1] * 1.0  # Should be √2
        assert result == pytest.approx(expected)
        assert result == pytest.approx(math.sqrt(2))

        # Point opposite to normal (outside)
        result = region.point((-1.0, -1.0))
        assert result == pytest.approx(-math.sqrt(2))

    def test_with_symbolic_expressions(self):
        """Test that the method works with sympy symbolic expressions."""
        region = RegionHalfspace(normal=(1.0, 0.0), anchor=(0.0, 0.0))

        # Create symbolic variables
        x = sp.Symbol("x", real=True)
        y = sp.Symbol("y", real=True)

        # Get symbolic result
        result = region.point((x, y))

        # Verify it's a symbolic expression
        assert isinstance(result, sp.Expr)

        # Evaluate at specific point
        evaluated = result.subs([(x, 3.0), (y, 2.0)])
        assert float(evaluated) == 3.0

    def test_unnormalized_normal_vector(self):
        """Test that unnormalized normal vectors work correctly."""
        # Normal not unit length: (2, 0)
        region = RegionHalfspace(normal=(2.0, 0.0), anchor=(0.0, 0.0))

        # Residual will be scaled by normal length
        result = region.point((1.0, 0.0))
        assert result == 2.0  # 2 * 1 = 2

        result = region.point((3.0, 5.0))
        assert result == 6.0  # 2 * 3 = 6

    def test_non_origin_anchor(self):
        """Test halfspace with anchor not at origin."""
        # Normal pointing right, anchor at (3, 4)
        region = RegionHalfspace(normal=(1.0, 0.0), anchor=(3.0, 4.0))

        # Point at anchor
        result = region.point((3.0, 4.0))
        assert result == 0.0

        # Point to the right of anchor (inside)
        result = region.point((5.0, 4.0))
        assert result == 2.0

        # Point to the left of anchor (outside)
        result = region.point((1.0, 4.0))
        assert result == -2.0

        # Y coordinate doesn't affect result with normal (1, 0)
        result = region.point((5.0, 10.0))
        assert result == 2.0


class TestRegionBall:
    """Test cases for RegionBall.point method."""

    def test_point_at_center(self):
        """Test that a point at the center has maximum positive residual."""
        region = RegionBall(center=(0.0, 0.0), radius=5.0)

        # Point at center
        result = region.point((0.0, 0.0))
        assert result == 5.0  # radius - 0

    def test_point_on_boundary(self):
        """Test that a point on the boundary returns zero residual."""
        region = RegionBall(center=(0.0, 0.0), radius=3.0)

        # Point on boundary (distance = radius)
        result = region.point((3.0, 0.0))
        assert result == pytest.approx(0.0)

        result = region.point((0.0, 3.0))
        assert result == pytest.approx(0.0)

        # Point on boundary at 45 degrees
        dist = 3.0 / math.sqrt(2)
        result = region.point((dist, dist))
        assert result == pytest.approx(0.0)

    def test_point_inside_ball(self):
        """Test that points inside the ball return positive residual."""
        region = RegionBall(center=(0.0, 0.0), radius=5.0)

        # Point at (3, 0) - distance is 3
        result = region.point((3.0, 0.0))
        assert result == pytest.approx(2.0)  # 5 - 3

        # Point at (3, 4) - distance is 5, but inside due to radius
        result = region.point((3.0, 4.0))
        assert result == pytest.approx(0.0)  # 5 - 5

    def test_point_outside_ball(self):
        """Test that points outside the ball return negative residual."""
        region = RegionBall(center=(0.0, 0.0), radius=2.0)

        # Point at (3, 0) - distance is 3
        result = region.point((3.0, 0.0))
        assert result == pytest.approx(-1.0)  # 2 - 3

        # Point at (3, 4) - distance is 5
        result = region.point((3.0, 4.0))
        assert result == pytest.approx(-3.0)  # 2 - 5

    def test_non_origin_center(self):
        """Test ball with center not at origin."""
        region = RegionBall(center=(2.0, 3.0), radius=4.0)

        # Point at center
        result = region.point((2.0, 3.0))
        assert result == pytest.approx(4.0)

        # Point on boundary: (2+4, 3) = (6, 3)
        result = region.point((6.0, 3.0))
        assert result == pytest.approx(0.0)

        # Point inside: (2+2, 3) = (4, 3), distance = 2
        result = region.point((4.0, 3.0))
        assert result == pytest.approx(2.0)  # 4 - 2

        # Point outside: (10, 3), distance = 8
        result = region.point((10.0, 3.0))
        assert result == pytest.approx(-4.0)  # 4 - 8

    def test_with_symbolic_expressions(self):
        """Test that the method works with sympy symbolic expressions."""
        region = RegionBall(center=(0.0, 0.0), radius=5.0)

        # Create symbolic variables
        x = sp.Symbol("x", real=True)
        y = sp.Symbol("y", real=True)

        # Get symbolic result
        result = region.point((x, y))

        # Verify it's a symbolic expression
        assert isinstance(result, sp.Expr)

        # Evaluate at specific point (3, 4) - distance is 5
        evaluated = result.subs([(x, 3.0), (y, 4.0)])
        assert float(evaluated) == pytest.approx(0.0)

        # Evaluate at origin
        evaluated = result.subs([(x, 0.0), (y, 0.0)])
        assert float(evaluated) == pytest.approx(5.0)

    def test_small_radius(self):
        """Test ball with small radius."""
        region = RegionBall(center=(1.0, 1.0), radius=0.5)

        # Point at center
        result = region.point((1.0, 1.0))
        assert result == pytest.approx(0.5)

        # Point just inside: (1.3, 1), distance = 0.3
        result = region.point((1.3, 1.0))
        assert result == pytest.approx(0.2)

        # Point just outside: (2, 1), distance = 1
        result = region.point((2.0, 1.0))
        assert result == pytest.approx(-0.5)

    def test_large_radius(self):
        """Test ball with large radius."""
        region = RegionBall(center=(0.0, 0.0), radius=100.0)

        # Point far from center but still inside
        result = region.point((60.0, 80.0))  # distance = 100
        assert result == pytest.approx(0.0)

        # Point inside
        result = region.point((30.0, 40.0))  # distance = 50
        assert result == pytest.approx(50.0)

    def test_pythagorean_triple(self):
        """Test with points at Pythagorean triple distances."""
        region = RegionBall(center=(0.0, 0.0), radius=10.0)

        # 3-4-5 triangle scaled to 6-8-10
        result = region.point((6.0, 8.0))  # distance = 10
        assert result == pytest.approx(0.0)

        # 5-12-13 triangle
        region2 = RegionBall(center=(0.0, 0.0), radius=13.0)
        result = region2.point((5.0, 12.0))  # distance = 13
        assert result == pytest.approx(0.0)
        result = region2.point((5.0, 12.0))  # distance = 13
        assert result == pytest.approx(0.0)
