import math

import pytest

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
    """Test cases for RegionHalfspace.point and .line methods."""

    def test_horizontal_halfspace(self):
        """Test halfspace with normal pointing right (horizontal boundary)."""
        # Halfspace with normal pointing right (1, 0), anchor at origin
        region = RegionHalfspace(normal=(1.0, 0.0), anchor=(0.0, 0.0))

        # Point on the boundary (at anchor) - on boundary counts as inside
        assert region.point((0.0, 0.0)) is True

        # Points to the right (inside)
        assert region.point((2.0, 0.0)) is True
        assert region.point((5.0, 3.0)) is True

        # Points to the left (outside)
        assert region.point((-3.0, 0.0)) is False
        assert region.point((-1.0, 5.0)) is False

    def test_vertical_halfspace(self):
        """Test halfspace with vertical boundary."""
        # Normal pointing up (0, 1), anchor at (2, 1)
        region = RegionHalfspace(normal=(0.0, 1.0), anchor=(2.0, 1.0))

        # Point above boundary (inside)
        assert region.point((2.0, 4.0)) is True

        # Point below boundary (outside)
        assert region.point((2.0, 0.0)) is False

        # Point on boundary
        assert region.point((5.0, 1.0)) is True

    def test_diagonal_halfspace(self):
        """Test halfspace with diagonal boundary."""
        # Normal at 45 degrees: (1/√2, 1/√2)
        normal = (1.0 / math.sqrt(2), 1.0 / math.sqrt(2))
        region = RegionHalfspace(normal=normal, anchor=(0.0, 0.0))

        # Point in direction of normal (inside)
        assert region.point((1.0, 1.0)) is True

        # Opposite to normal (outside)
        assert region.point((-1.0, -1.0)) is False

    def test_non_origin_anchor(self):
        """Test halfspace with anchor not at origin."""
        # Normal pointing right, anchor at (3, 4)
        region = RegionHalfspace(normal=(1.0, 0.0), anchor=(3.0, 4.0))

        assert region.point((3.0, 4.0)) is True  # Point at anchor (on boundary)
        assert region.point((5.0, 4.0)) is True  # Point to the right of anchor (inside)
        assert (
            region.point((1.0, 4.0)) is False
        )  # Point to the left of anchor (outside)
        assert region.point((5.0, 10.0)) is True  # Y coordinate doesn't affect result

    def test_line_simple(self):
        """Test line segment collision with halfspace."""
        # Normal pointing up, anchor at origin
        region = RegionHalfspace(normal=(0.0, 1.0), anchor=(0.0, 0.0))

        # Both endpoints inside
        assert region.line(((1.0, 1.0), (2.0, 2.0))) is True

        # Both endpoints outside
        assert region.line(((1.0, -1.0), (2.0, -2.0))) is False

        # One inside, one outside (crosses boundary)
        assert region.line(((0.0, -1.0), (0.0, 1.0))) is True

        # Both on boundary
        assert region.line(((0.0, 0.0), (5.0, 0.0))) is True


class TestRegionBall:
    """Test cases for RegionBall.point and .line methods."""

    def test_ball_at_origin(self):
        """Test ball centered at origin with various points."""
        region = RegionBall(center=(0.0, 0.0), radius=5.0)

        # Point at center
        assert region.point((0.0, 0.0)) is True

        # Points inside
        assert region.point((3.0, 0.0)) is True
        assert region.point((3.0, 4.0)) is True  # On boundary (distance = 5)

    def test_ball_boundary_points(self):
        """Test points on the boundary of a ball."""
        region = RegionBall(center=(0.0, 0.0), radius=3.0)

        # Points on boundary (distance = radius) - on boundary counts as inside
        assert region.point((3.0, 0.0)) is True
        assert region.point((0.0, 3.0)) is True

        # Point on boundary at 45 degrees
        dist = 3.0 / math.sqrt(2)
        assert region.point((dist, dist)) is True

    def test_ball_outside_points(self):
        """Test points outside a ball."""
        region = RegionBall(center=(0.0, 0.0), radius=2.0)

        # Points outside
        assert region.point((3.0, 0.0)) is False
        assert region.point((3.0, 4.0)) is False

    def test_ball_non_origin_center(self):
        """Test ball with center not at origin."""
        region = RegionBall(center=(2.0, 3.0), radius=4.0)

        assert region.point((2.0, 3.0)) is True  # At center
        assert region.point((6.0, 3.0)) is True  # On boundary: (2+4, 3)
        assert region.point((4.0, 3.0)) is True  # Inside
        assert region.point((10.0, 3.0)) is False  # Outside

    def test_small_radius(self):
        """Test ball with small radius."""
        region = RegionBall(center=(1.0, 1.0), radius=0.5)

        assert region.point((1.0, 1.0)) is True  # At center
        assert region.point((1.3, 1.0)) is True  # Inside
        assert region.point((2.0, 1.0)) is False  # Outside

    def test_large_radius(self):
        """Test ball with large radius."""
        region = RegionBall(center=(0.0, 0.0), radius=100.0)

        assert region.point((60.0, 80.0)) is True  # On boundary
        assert region.point((30.0, 40.0)) is True  # Inside

    def test_pythagorean_triple(self):
        """Test with points at Pythagorean triple distances."""
        region = RegionBall(center=(0.0, 0.0), radius=10.0)
        assert region.point((6.0, 8.0)) is True  # On boundary (3-4-5 scaled to 6-8-10)

        region2 = RegionBall(center=(0.0, 0.0), radius=13.0)
        assert region2.point((5.0, 12.0)) is True  # On boundary (5-12-13 triangle)

    def test_line_simple(self):
        """Test line segment with simple cases."""
        region = RegionBall(center=(0.0, 0.0), radius=5.0)

        assert region.line(((1.0, 1.0), (2.0, 2.0))) is True  # Both inside
        assert region.line(((2.0, 0.0), (8.0, 0.0))) is True  # One inside, one outside
        assert region.line(((0.0, 0.0), (0.0, 0.0))) is True  # Zero-length at center
        assert (
            region.line(((10.0, 0.0), (0.0, 10.0))) is False
        )  # Both outside, no intersection
        assert region.line(((5.0, 0.0), (0.0, 5.0))) is True  # Both on boundary

    def test_line_segment(self):
        """Test line segment with more difficult cases."""
        region = RegionBall(center=(0.0, 0.0), radius=3.0)

        # Cases where both points are outside, but the line segment intersects the ball:
        assert region.line(((-3.0, 1.0), (3.0, 1.0))) is True
        assert region.line(((-3.0, 1.0), (1.0, -3.0))) is True

        # Cases where both points are outside, but the line segment does not intersect the ball:
        assert region.line(((4.0, 1.0), (3.5, 1.0))) is False
        assert region.line(((4.0, 1.0), (1.0, 4.0))) is False

    def test_line_non_origin_ball(self):
        """Test line segments with ball not centered at origin."""
        region = RegionBall(center=(3.0, 4.0), radius=2.0)

        assert region.line(((3.5, 4.0), (3.0, 4.5))) is True  # Both inside
        assert region.line(((3.0, 4.0), (6.0, 4.0))) is True  # One inside, one outside


class TestRegionRectangle:
    """Test cases for RegionRectangle.point and .line methods."""

    def test_rectangle_at_origin(self):
        """Test rectangle at origin with various points."""
        region = RegionRectangle(left=0.0, right=10.0, bottom=0.0, top=10.0)

        # Point at center
        assert region.point((5.0, 5.0)) is True

        # Points inside near edges
        assert region.point((1.0, 5.0)) is True
        assert region.point((9.0, 5.0)) is True
        assert region.point((5.0, 1.0)) is True
        assert region.point((5.0, 9.0)) is True

        # Points on each edge (on boundary counts as inside)
        assert region.point((0.0, 5.0)) is True
        assert region.point((10.0, 5.0)) is True
        assert region.point((5.0, 0.0)) is True
        assert region.point((5.0, 10.0)) is True

        # All four corners
        assert region.point((0.0, 0.0)) is True
        assert region.point((10.0, 0.0)) is True
        assert region.point((0.0, 10.0)) is True
        assert region.point((10.0, 10.0)) is True

        # Points outside each edge
        assert region.point((-2.0, 5.0)) is False
        assert region.point((13.0, 5.0)) is False
        assert region.point((5.0, -4.0)) is False
        assert region.point((5.0, 15.0)) is False

        # Points outside corners
        assert region.point((-2.0, -3.0)) is False
        assert region.point((12.0, 15.0)) is False

    def test_rectangle_non_origin(self):
        """Test rectangle not centered at origin."""
        region = RegionRectangle(left=-5.0, right=5.0, bottom=-3.0, top=3.0)

        assert region.point((0.0, 0.0)) is True  # Center
        assert region.point((-5.0, 0.0)) is True  # On left edge
        assert region.point((-7.0, 0.0)) is False  # Outside left

    def test_rectangle_small(self):
        """Test small rectangle."""
        region = RegionRectangle(left=1.0, right=2.0, bottom=1.0, top=2.0)

        assert region.point((1.5, 1.5)) is True  # Center
        assert region.point((0.0, 1.5)) is False  # Outside

    def test_rectangle_non_square(self):
        """Test wide and tall (non-square) rectangles."""
        # Wide rectangle
        region_wide = RegionRectangle(left=0.0, right=20.0, bottom=0.0, top=5.0)
        assert region_wide.point((10.0, 2.5)) is True
        assert region_wide.point((2.0, 2.5)) is True

        # Tall rectangle
        region_tall = RegionRectangle(left=0.0, right=5.0, bottom=0.0, top=20.0)
        assert region_tall.point((2.5, 10.0)) is True
        assert region_tall.point((2.5, 3.0)) is True

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

        assert region.point((-6.0, -4.0)) is True  # Inside
        assert region.point((-12.0, -4.0)) is False  # Outside left
        assert region.point((0.0, -4.0)) is False  # Outside right

    def test_line_simple(self):
        """Test line segment with simple cases."""
        region = RegionRectangle(left=0.0, right=4.0, bottom=0.0, top=4.0)

        assert region.line(((1.0, 1.0), (2.0, 2.0))) is True  # Both inside
        assert region.line(((2.0, 2.0), (5.0, 2.0))) is True  # One inside, one outside
        assert (
            region.line(((5.0, 5.0), (6.0, 6.0))) is False
        )  # Both outside, no intersection
        assert (
            region.line(((0.0, 0.0), (0.0, 0.0))) is True
        )  # Zero-length at corner (on boundary)

    def test_line_segment(self):
        """Test line segment collision with rectangles."""
        region = RegionRectangle(left=0.0, right=4.0, bottom=0.0, top=4.0)

        # Both endpoints outside, horizontal segment crosses rectangle
        assert region.line(((-1.0, 2.0), (5.0, 2.0))) is True

        # Both endpoints outside, vertical segment crosses rectangle
        assert region.line(((2.0, -1.0), (2.0, 5.0))) is True

        # Diagonal segment through rectangle
        assert region.line(((-1.0, -1.0), (5.0, 5.0))) is True

        # Segment on boundary (left edge)
        assert region.line(((0.0, 1.0), (0.0, 3.0))) is True

        # Segment parallel to rectangle but outside
        assert region.line(((5.0, 0.0), (5.0, 4.0))) is False

    def test_line_non_origin_rectangle(self):
        """Test line segments with rectangle not at origin."""
        region = RegionRectangle(left=-2.0, right=2.0, bottom=-2.0, top=2.0)

        assert region.line(((0.0, 0.0), (1.0, 1.0))) is True  # Both inside
        assert region.line(((-3.0, 0.0), (3.0, 0.0))) is True  # Crosses through center
        assert region.line(((3.0, 3.0), (4.0, 4.0))) is False  # Both outside
