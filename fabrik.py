#!python3

"""FABRIK (Forward And Backward Reaching Inverse Kinematics) solver.

FABRIK is a heuristic iterative method that solves IK by alternating between:
1. Forward reaching: Move end effector to target, then adjust joints backward to base
2. Backward reaching: Fix base position, then adjust joints forward to end effector

This implementation follows the interface pattern from symbolic.py but uses
numpy arrays for numerical operations.
"""

import time
from typing import Literal

import numpy as np

from datamodel import (
    DesiredPosition,
    IKReturn,
    IKSolver,
    Region,
    RegionBall,
    RegionHalfspace,
    RegionRectangle,
    RobotModel,
    RobotPosition,
    RobotState,
    WorldModel,
)


# Numpy-based region collision detection helpers for FABRIK


class NumpyRegionHalfspace:
    """Numpy collision detection for RegionHalfspace."""

    def __init__(self, region: RegionHalfspace):
        self.normal = np.array(region.normal, dtype=np.float64)
        self.normal = self.normal / np.linalg.norm(self.normal)  # Normalize
        self.anchor = np.array(region.anchor, dtype=np.float64)

    def point_residual(self, point: np.ndarray) -> float:
        """Compute signed distance. Positive = inside (violation)."""
        return float(np.dot(self.normal, point - self.anchor).item())

    def project_point(self, point: np.ndarray) -> np.ndarray:
        """Project point outside the halfspace if inside."""
        residual = self.point_residual(point)
        if residual > 0:
            # Push point outside along normal direction
            return point - self.normal * (residual + 1e-4)
        return point

    def line_intersects(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """Check if line segment intersects the halfspace interior."""
        r1 = self.point_residual(p1)
        r2 = self.point_residual(p2)
        # Both inside or one inside
        return r1 > 0 or r2 > 0


class NumpyRegionBall:
    """Numpy collision detection for RegionBall."""

    def __init__(self, region: RegionBall):
        self.center = np.array(region.center, dtype=np.float64)
        self.radius = region.radius

    def point_residual(self, point: np.ndarray) -> float:
        """Compute signed distance. Positive = inside (violation)."""
        dist = np.linalg.norm(point - self.center)
        return self.radius - dist

    def project_point(self, point: np.ndarray) -> np.ndarray:
        """Project point outside the ball if inside."""
        diff = point - self.center
        dist = np.linalg.norm(diff)
        if dist < self.radius:
            if dist < 1e-10:
                # Point at center, push in arbitrary direction
                return self.center + np.array([self.radius + 1e-4, 0.0])
            # Push point to surface plus small margin
            return self.center + diff / dist * (self.radius + 1e-4)
        return point

    def line_intersects(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """Check if line segment intersects the ball."""
        # Check endpoints
        if self.point_residual(p1) > 0 or self.point_residual(p2) > 0:
            return True
        # Check closest point on segment to center
        d = p2 - p1
        length_sq = float(np.dot(d, d))
        if length_sq < 1e-10:
            return False
        t = float(np.clip(np.dot(self.center - p1, d) / length_sq, 0.0, 1.0))
        closest = p1 + t * d
        return bool(np.linalg.norm(closest - self.center) < self.radius)


class NumpyRegionRectangle:
    """Numpy collision detection for RegionRectangle."""

    def __init__(self, region: RegionRectangle):
        self.left = region.left
        self.right = region.right
        self.bottom = region.bottom
        self.top = region.top

    def point_residual(self, point: np.ndarray) -> float:
        """Compute signed distance to boundary. Positive = inside (violation)."""
        x, y = point
        # Distance to each boundary (positive when inside)
        dist_left = x - self.left
        dist_right = self.right - x
        dist_bottom = y - self.bottom
        dist_top = self.top - y
        # Minimum determines if inside and by how much
        return min(dist_left, dist_right, dist_bottom, dist_top)

    def project_point(self, point: np.ndarray) -> np.ndarray:
        """Project point outside the rectangle if inside."""
        x, y = point
        if not (self.left < x < self.right and self.bottom < y < self.top):
            return point  # Already outside

        # Find closest edge and push outside
        dist_left = x - self.left
        dist_right = self.right - x
        dist_bottom = y - self.bottom
        dist_top = self.top - y

        min_dist = min(dist_left, dist_right, dist_bottom, dist_top)
        margin = 1e-4

        if min_dist == dist_left:
            return np.array([self.left - margin, y])
        elif min_dist == dist_right:
            return np.array([self.right + margin, y])
        elif min_dist == dist_bottom:
            return np.array([x, self.bottom - margin])
        else:
            return np.array([x, self.top + margin])

    def line_intersects(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """Check if line segment intersects the rectangle."""
        # Check if either endpoint is inside
        if self.point_residual(p1) > 0 or self.point_residual(p2) > 0:
            return True

        # Check intersection with each edge using parametric line intersection
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1

        def intersects_vertical_edge(edge_x: float) -> bool:
            if abs(dx) < 1e-10:
                return False
            t = (edge_x - x1) / dx
            if 0 <= t <= 1:
                y_at_t = y1 + t * dy
                return self.bottom <= y_at_t <= self.top
            return False

        def intersects_horizontal_edge(edge_y: float) -> bool:
            if abs(dy) < 1e-10:
                return False
            t = (edge_y - y1) / dy
            if 0 <= t <= 1:
                x_at_t = x1 + t * dx
                return self.left <= x_at_t <= self.right
            return False

        return (
            intersects_vertical_edge(self.left)
            or intersects_vertical_edge(self.right)
            or intersects_horizontal_edge(self.bottom)
            or intersects_horizontal_edge(self.top)
        )


NumpyRegion = NumpyRegionHalfspace | NumpyRegionBall | NumpyRegionRectangle


def make_numpy_region(region: Region) -> NumpyRegion:
    """Factory function to create the appropriate numpy region helper."""
    if isinstance(region, RegionHalfspace):
        return NumpyRegionHalfspace(region)
    elif isinstance(region, RegionBall):
        return NumpyRegionBall(region)
    elif isinstance(region, RegionRectangle):
        return NumpyRegionRectangle(region)
    else:
        raise TypeError(f"Unknown region type: {type(region)}")


class IKFabrik(IKSolver):
    """Implements FABRIK inverse kinematics solver using numpy."""

    def __init__(
        self,
        model: RobotModel,
        world: WorldModel | None,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        collision_geometry: Literal["line", "point"] = "line",
    ) -> None:
        self.model = model
        self.nogo = world.nogo if world else None
        self.n_joints = len(model.link_lengths)
        self.collision_geometry = collision_geometry

        # Optimization parameters
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Store link lengths as numpy array
        self.link_lengths = np.array(model.link_lengths, dtype=np.float64)
        self.joint_origins = np.array(model.joint_origins, dtype=np.float64)

        # Process joint limits from model
        # joint_limits is a tuple of (min, max) or None for each joint
        self.joint_bounds: list[tuple[float, float] | None] = []
        if model.joint_limits:
            for limit in model.joint_limits:
                if limit is not None:
                    self.joint_bounds.append((limit[0], limit[1]))
                else:
                    self.joint_bounds.append(None)
        # Pad with None if joint_limits is shorter than number of joints
        while len(self.joint_bounds) < self.n_joints:
            self.joint_bounds.append(None)

        # Total reach of the robot
        self.total_reach = float(np.sum(self.link_lengths))

        # Pre-build numpy region helpers if nogo zones exist
        self.numpy_regions: list[NumpyRegion] | None = None
        if self.nogo:
            self.numpy_regions = [make_numpy_region(r) for r in self.nogo]

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

    def _clamp_joint_angles(self, angles: np.ndarray) -> np.ndarray:
        """Clamp joint angles to their limits.

        Args:
            angles: Joint angles as (n_joints,) array.

        Returns:
            Clamped joint angles as (n_joints,) array.
        """
        clamped = angles.copy()
        for i, bound in enumerate(self.joint_bounds):
            if bound is not None:
                clamped[i] = np.clip(clamped[i], bound[0], bound[1])
        return clamped

    def _apply_joint_limits(self, positions: np.ndarray) -> np.ndarray:
        """Apply joint limits by converting to angles, clamping, and converting back.

        Args:
            positions: Joint positions as (n_joints + 1, 2) array.

        Returns:
            Updated joint positions with joint limits enforced.
        """
        # Check if any joint limits are defined
        if not any(b is not None for b in self.joint_bounds):
            return positions

        # Convert positions to angles
        angles = self._positions_to_angles(positions)

        # Clamp angles to limits
        clamped_angles = self._clamp_joint_angles(angles)

        # If no changes, return original positions
        if np.allclose(angles, clamped_angles):
            return positions

        # Convert back to positions
        return self._forward_kinematics(clamped_angles)

    def _project_point_from_regions(self, point: np.ndarray) -> np.ndarray:
        """Project a point outside all nogo regions.

        Args:
            point: Point to project (2,).

        Returns:
            Projected point outside all regions.
        """
        if not self.numpy_regions:
            return point

        # Iteratively project out of each region
        projected = point.copy()
        for _ in range(10):  # Max iterations to handle overlapping regions
            moved = False
            for region in self.numpy_regions:
                if region.point_residual(projected) > 0:
                    projected = region.project_point(projected)
                    moved = True
            if not moved:
                break

        return projected

    def _check_link_collision(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """Check if a link segment collides with any nogo region.

        Args:
            p1: Start point of link.
            p2: End point of link.

        Returns:
            True if collision detected.
        """
        if not self.numpy_regions:
            return False

        for region in self.numpy_regions:
            if region.line_intersects(p1, p2):
                return True
        return False

    def _adjust_link_for_collision(
        self, p_fixed: np.ndarray, p_free: np.ndarray, link_length: float
    ) -> np.ndarray:
        """Adjust a free endpoint to avoid collision while maintaining link length.

        Uses a search strategy: if direct path collides, try rotating the link
        to find a collision-free configuration.

        Args:
            p_fixed: Fixed endpoint of the link.
            p_free: Free endpoint to adjust.
            link_length: Length of the link.

        Returns:
            Adjusted position for the free endpoint.
        """
        if not self.numpy_regions:
            return p_free

        # First, project the free point out of any regions
        p_free_projected = self._project_point_from_regions(p_free)

        # Recompute direction and place at correct distance
        direction = p_free_projected - p_fixed
        dist = np.linalg.norm(direction)
        if dist > 1e-10:
            direction = direction / dist
        else:
            direction = np.array([1.0, 0.0])

        candidate = p_fixed + direction * link_length

        # Check if this creates a collision-free link
        if not self._check_link_collision(p_fixed, candidate):
            return candidate

        # If still colliding, search for a better angle
        # Try rotating in small increments
        original_angle = np.arctan2(direction[1], direction[0])

        for delta in np.linspace(0.1, np.pi, 20):
            for sign in [1, -1]:
                test_angle = original_angle + sign * delta
                test_dir = np.array([np.cos(test_angle), np.sin(test_angle)])
                test_point = p_fixed + test_dir * link_length

                # Check both point and line collision
                if (
                    self._project_point_from_regions(test_point) is test_point
                    or np.allclose(
                        self._project_point_from_regions(test_point), test_point
                    )
                ) and not self._check_link_collision(p_fixed, test_point):
                    return test_point

        # Fallback: return the projected candidate even if not perfect
        return self._project_point_from_regions(candidate)

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
            candidate = positions[i] + direction * self.link_lengths[i]

            # Apply collision constraints
            if self.numpy_regions:
                if self.collision_geometry == "point":
                    # Only project the joint position out of regions
                    candidate = self._project_point_from_regions(candidate)
                    # Re-adjust to maintain link length
                    direction = candidate - positions[i]
                    dist = np.linalg.norm(direction)
                    if dist > 1e-10:
                        direction = direction / dist
                        candidate = positions[i] + direction * self.link_lengths[i]
                else:  # line
                    # Adjust link to avoid collision
                    candidate = self._adjust_link_for_collision(
                        positions[i], candidate, self.link_lengths[i]
                    )

            positions[i + 1] = candidate

        # Apply joint limits after forward pass
        positions = self._apply_joint_limits(positions)

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
    ) -> IKReturn:
        """Solve IK for the desired position using FABRIK.

        Args:
            state: Current robot state.
            desired: Desired end effector position and optional angle.

        Returns:
            IKReturn containing the solution state and profiling information.
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

        initial_loss = np.linalg.norm(positions[-1] - target)
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

        final_loss = float(np.linalg.norm(positions[-1] - target))

        return IKReturn(
            state=result_state,
            solve_time_ms=(end_time - start_time) * 1000,
            iterations=iterations_completed,
            converged=converged,
            initial_loss=initial_loss,
            final_loss=final_loss,
            position_error=final_loss,
        )
