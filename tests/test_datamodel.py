# pyright: reportArgumentType=false

import math

import pytest
import sympy as sp

from datamodel import (
    RegionBall,
    RegionHalfspace,
    RegionRectangle,
    RobotModel,
    RobotPosition,
)


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

    def test_horizontal_halfspace(self):
        """Test halfspace with normal pointing right (horizontal boundary)."""
        # Halfspace with normal pointing right (1, 0), anchor at origin
        region = RegionHalfspace(normal=(1.0, 0.0), anchor=(0.0, 0.0))

        # Point on the boundary (at anchor)
        assert region.point((0.0, 0.0)) == 0.0

        # Points to the right (inside)
        assert region.point((2.0, 0.0)) == 2.0
        assert region.point((5.0, 3.0)) == 5.0  # Only x-component matters

        # Points to the left (outside)
        assert region.point((-3.0, 0.0)) == -3.0
        assert region.point((-1.0, 5.0)) == -1.0

    def test_vertical_halfspace(self):
        """Test halfspace with vertical boundary."""
        # Normal pointing up (0, 1), anchor at (2, 1)
        region = RegionHalfspace(normal=(0.0, 1.0), anchor=(2.0, 1.0))

        # Point above boundary (inside)
        assert region.point((2.0, 4.0)) == 3.0

        # Point below boundary (outside)
        assert region.point((2.0, 0.0)) == -1.0

        # Point on boundary (x doesn't matter)
        assert region.point((5.0, 1.0)) == 0.0

    def test_diagonal_halfspace(self):
        """Test halfspace with diagonal boundary."""
        # Normal at 45 degrees: (1/√2, 1/√2)
        normal = (1.0 / math.sqrt(2), 1.0 / math.sqrt(2))
        region = RegionHalfspace(normal=normal, anchor=(0.0, 0.0))

        # Point in direction of normal (inside)
        expected = normal[0] * 1.0 + normal[1] * 1.0  # Should be √2
        assert region.point((1.0, 1.0)) == pytest.approx(expected)
        assert region.point((1.0, 1.0)) == pytest.approx(math.sqrt(2))

        assert region.point((-1.0, -1.0)) == pytest.approx(
            -math.sqrt(2)
        )  # Opposite to normal (outside)

    def test_with_symbolic_expressions(self):
        """Test that the method works with sympy symbolic expressions."""
        region = RegionHalfspace(normal=(1.0, 0.0), anchor=(0.0, 0.0))

        # Create symbolic variables
        x = sp.Symbol("x", real=True)
        y = sp.Symbol("y", real=True)

        result = region.point((x, y))
        assert isinstance(result, sp.Expr)  # Verify it's a symbolic expression
        assert float(result.subs([(x, 3.0), (y, 2.0)])) == 3.0

    def test_unnormalized_normal_vector(self):
        """Test that unnormalized normal vectors work correctly."""
        # Normal not unit length: (2, 0)
        region = RegionHalfspace(normal=(2.0, 0.0), anchor=(0.0, 0.0))

        # Residual will be scaled by normal length
        assert region.point((1.0, 0.0)) == 2.0  # 2 * 1 = 2
        assert region.point((3.0, 5.0)) == 6.0  # 2 * 3 = 6

    def test_non_origin_anchor(self):
        """Test halfspace with anchor not at origin."""
        # Normal pointing right, anchor at (3, 4)
        region = RegionHalfspace(normal=(1.0, 0.0), anchor=(3.0, 4.0))

        assert region.point((3.0, 4.0)) == 0.0  # Point at anchor
        assert region.point((5.0, 4.0)) == 2.0  # Point to the right of anchor (inside)
        assert region.point((1.0, 4.0)) == -2.0  # Point to the left of anchor (outside)
        assert region.point((5.0, 10.0)) == 2.0  # Y coordinate doesn't affect result


class TestRegionBall:
    """Test cases for RegionBall.point method."""

    def test_ball_at_origin(self):
        """Test ball centered at origin with various points."""
        region = RegionBall(center=(0.0, 0.0), radius=5.0)

        # Point at center
        assert region.point((0.0, 0.0)) == 5.0  # radius - 0

        # Points inside
        assert region.point((3.0, 0.0)) == pytest.approx(2.0)  # 5 - 3
        assert region.point((3.0, 4.0)) == pytest.approx(0.0)

    def test_ball_boundary_points(self):
        """Test points on the boundary of a ball."""
        region = RegionBall(center=(0.0, 0.0), radius=3.0)

        # Points on boundary (distance = radius)
        assert region.point((3.0, 0.0)) == pytest.approx(0.0)
        assert region.point((0.0, 3.0)) == pytest.approx(0.0)

        # Point on boundary at 45 degrees
        dist = 3.0 / math.sqrt(2)
        assert region.point((dist, dist)) == pytest.approx(0.0)

    def test_ball_outside_points(self):
        """Test points outside a ball."""
        region = RegionBall(center=(0.0, 0.0), radius=2.0)

        # Points outside
        assert region.point((3.0, 0.0)) == pytest.approx(-1.0)  # distance 3: 2 - 3 = -1
        assert region.point((3.0, 4.0)) == pytest.approx(-3.0)  # distance 5: 2 - 5 = -3

    def test_ball_non_origin_center(self):
        """Test ball with center not at origin."""
        region = RegionBall(center=(2.0, 3.0), radius=4.0)

        assert region.point((2.0, 3.0)) == pytest.approx(4.0)  # At center
        assert region.point((6.0, 3.0)) == pytest.approx(0.0)  # On boundary: (2+4, 3)
        assert region.point((4.0, 3.0)) == pytest.approx(2.0)
        assert region.point((10.0, 3.0)) == pytest.approx(-4.0)

    def test_with_symbolic_expressions(self):
        """Test that the method works with sympy symbolic expressions."""
        region = RegionBall(center=(0.0, 0.0), radius=5.0)

        # Create symbolic variables
        x = sp.Symbol("x", real=True)
        y = sp.Symbol("y", real=True)

        result = region.point((x, y))
        assert isinstance(result, sp.Expr)  # Verify it's a symbolic expression

        assert float(result.subs([(x, 3.0), (y, 4.0)])) == pytest.approx(
            0.0
        )  # Point (3, 4) - distance is 5
        assert float(result.subs([(x, 0.0), (y, 0.0)])) == pytest.approx(
            5.0
        )  # At origin

    def test_small_radius(self):
        """Test ball with small radius."""
        region = RegionBall(center=(1.0, 1.0), radius=0.5)

        assert region.point((1.0, 1.0)) == pytest.approx(0.5)  # At center
        assert region.point((1.3, 1.0)) == pytest.approx(
            0.2
        )  # Inside: distance 0.3, residual 0.5-0.3
        assert region.point((2.0, 1.0)) == pytest.approx(
            -0.5
        )  # Outside: distance 1, residual 0.5-1

    def test_large_radius(self):
        """Test ball with large radius."""
        region = RegionBall(center=(0.0, 0.0), radius=100.0)

        assert region.point((60.0, 80.0)) == pytest.approx(
            0.0
        )  # On boundary: distance 100
        assert region.point((30.0, 40.0)) == pytest.approx(50.0)  # Inside: distance 50

    def test_pythagorean_triple(self):
        """Test with points at Pythagorean triple distances."""
        region = RegionBall(center=(0.0, 0.0), radius=10.0)
        assert region.point((6.0, 8.0)) == pytest.approx(0.0)  # 3-4-5 scaled to 6-8-10

        region2 = RegionBall(center=(0.0, 0.0), radius=13.0)
        assert region2.point((5.0, 12.0)) == pytest.approx(0.0)  # 5-12-13 triangle

    def test_line_simple(self):
        """Test line segment with simple cases."""
        region = RegionBall(center=(0.0, 0.0), radius=5.0)

        assert float(region.line((1.0, 1.0), (2.0, 2.0))) > 0.0  # Both inside
        assert float(region.line((2, 0), (8, 0))) == pytest.approx(3.0)
        assert float(region.line((0.0, 0.0), (0.0, 0.0))) == pytest.approx(10.0)
        assert float(region.line((10.0, 0.0), (0.0, 10.0))) == pytest.approx(0.0)
        assert float(region.line((5.0, 0.0), (0.0, 5.0))) > 0  # Both on boundary

    def test_line_non_origin_ball(self):
        """Test line segments with ball not centered at origin."""
        region = RegionBall(center=(3.0, 4.0), radius=2.0)

        # Both endpoints inside
        assert float(region.line((3.5, 4.0), (3.0, 4.5))) == pytest.approx(
            3.0
        )  # Both 0.5 from center

        # One inside, one outside
        assert float(region.line((3.0, 4.0), (6.0, 4.0))) == pytest.approx(
            2.0
        )  # Center and outside

    def test_line_with_symbolic_expressions(self):
        """Test line method with symbolic expressions."""
        region = RegionBall(center=(0.0, 0.0), radius=5.0)

        # Create symbolic variables
        x1 = sp.Symbol("x1", real=True)
        y1 = sp.Symbol("y1", real=True)
        x2 = sp.Symbol("x2", real=True)
        y2 = sp.Symbol("y2", real=True)

        result = region.line((x1, y1), (x2, y2))
        assert isinstance(result, sp.Expr)

        assert float(
            result.subs([(x1, 1.0), (y1, 1.0), (x2, 2.0), (y2, 2.0)])
        ) == pytest.approx(5.758, abs=0.01)  # Both points inside
        assert (
            float(result.subs([(x1, 10.0), (y1, 0.0), (x2, 0.0), (y2, 10.0)])) == 0.0
        )  # Both points outside


class TestRegionRectangle:
    """Test cases for RegionRectangle.point method."""

    def test_rectangle_at_origin(self):
        """Test rectangle at origin with various points."""
        region = RegionRectangle(left=0.0, right=10.0, bottom=0.0, top=10.0)

        # Point at center
        assert float(region.point((5.0, 5.0))) == 5.0  # All edges 5 units away

        # Points inside near edges
        assert float(region.point((1.0, 5.0))) == 1.0  # Near left
        assert float(region.point((9.0, 5.0))) == 1.0  # Near right
        assert float(region.point((5.0, 1.0))) == 1.0  # Near bottom
        assert float(region.point((5.0, 9.0))) == 1.0  # Near top

        # Points on each edge
        assert float(region.point((0.0, 5.0))) == 0.0  # Left edge
        assert float(region.point((10.0, 5.0))) == 0.0  # Right edge
        assert float(region.point((5.0, 0.0))) == 0.0  # Bottom edge
        assert float(region.point((5.0, 10.0))) == 0.0  # Top edge

        # All four corners
        assert float(region.point((0.0, 0.0))) == 0.0  # Bottom-left
        assert float(region.point((10.0, 0.0))) == 0.0  # Bottom-right
        assert float(region.point((0.0, 10.0))) == 0.0  # Top-left
        assert float(region.point((10.0, 10.0))) == 0.0  # Top-right

        # Points outside each edge
        assert float(region.point((-2.0, 5.0))) == -2.0  # Outside left
        assert float(region.point((13.0, 5.0))) == -3.0  # Outside right
        assert float(region.point((5.0, -4.0))) == -4.0  # Outside bottom
        assert float(region.point((5.0, 15.0))) == -5.0  # Outside top

        # Points outside corners
        assert (
            float(region.point((-2.0, -3.0))) == -3.0
        )  # Bottom-left: min(-2, -3) = -3
        assert float(region.point((12.0, 15.0))) == -5.0  # Top-right: min(-2, -5) = -5

    def test_rectangle_non_origin(self):
        """Test rectangle not centered at origin."""
        region = RegionRectangle(left=-5.0, right=5.0, bottom=-3.0, top=3.0)

        assert (
            float(region.point((0.0, 0.0))) == 3.0
        )  # Center: closest edge is top/bottom at 3
        assert float(region.point((-5.0, 0.0))) == 0.0  # On left edge
        assert float(region.point((-7.0, 0.0))) == -2.0  # Outside left

    def test_rectangle_small(self):
        """Test small rectangle."""
        region = RegionRectangle(left=1.0, right=2.0, bottom=1.0, top=2.0)

        assert float(region.point((1.5, 1.5))) == 0.5  # Center
        assert float(region.point((0.0, 1.5))) == -1.0  # Outside

    def test_rectangle_non_square(self):
        """Test wide and tall (non-square) rectangles."""
        # Wide rectangle
        region_wide = RegionRectangle(left=0.0, right=20.0, bottom=0.0, top=5.0)

        assert (
            float(region_wide.point((10.0, 2.5))) == 2.5
        )  # Closer to top/bottom (2.5) than sides (10)
        assert (
            float(region_wide.point((2.0, 2.5))) == 2.0
        )  # Closer to left (2) than top/bottom (2.5)

        # Tall rectangle
        region_tall = RegionRectangle(left=0.0, right=5.0, bottom=0.0, top=20.0)

        assert (
            float(region_tall.point((2.5, 10.0))) == 2.5
        )  # Closer to left/right (2.5) than top/bottom (10)
        assert (
            float(region_tall.point((2.5, 3.0))) == 2.5
        )  # Closer to bottom (3) than sides (2.5)

    def test_with_symbolic_expressions(self):
        """Test that the method works with sympy symbolic expressions."""
        region = RegionRectangle(left=0.0, right=10.0, bottom=0.0, top=10.0)

        # Create symbolic variables
        x = sp.Symbol("x", real=True)
        y = sp.Symbol("y", real=True)

        # Get symbolic result
        result = region.point((x, y))

        assert isinstance(result, sp.Expr)  # Verify it's a symbolic expression

        assert (
            float(result.subs([(x, 5.0), (y, 5.0)])) == 5.0
        )  # Evaluate at center (5, 5)
        assert (
            float(result.subs([(x, 15.0), (y, 5.0)])) == -5.0
        )  # Evaluate outside (15, 5)

    def test_invalid_boundaries(self):
        """Test that invalid boundary specifications raise errors."""
        # Left >= right should raise ValueError
        with pytest.raises(ValueError, match="left.*must be less than right"):
            RegionRectangle(left=10.0, right=5.0, bottom=0.0, top=10.0)

        # Bottom >= top should raise ValueError
        with pytest.raises(ValueError, match="bottom.*must be less than top"):
            RegionRectangle(left=0.0, right=10.0, bottom=10.0, top=5.0)

        # Equal boundaries should also fail
        with pytest.raises(ValueError, match="left.*must be less than right"):
            RegionRectangle(left=5.0, right=5.0, bottom=0.0, top=10.0)

        with pytest.raises(ValueError, match="bottom.*must be less than top"):
            RegionRectangle(left=0.0, right=10.0, bottom=5.0, top=5.0)

    def test_rectangle_negative_coordinates(self):
        """Test rectangle with negative coordinates."""
        region = RegionRectangle(left=-10.0, right=-2.0, bottom=-8.0, top=-1.0)

        assert (
            float(region.point((-6.0, -4.0))) == 3.0
        )  # Inside: distances left=4, right=4, bottom=3, top=3
        assert float(region.point((-12.0, -4.0))) == -2.0  # Outside left
        assert float(region.point((0.0, -4.0))) == -2.0  # Outside right
