#!python3

import time
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch

from datamodel import (
    DesiredPosition,
    Region,
    RegionBall,
    RegionHalfspace,
    RegionRectangle,
    RobotModel,
    RobotPosition,
    RobotState,
    WorldModel,
)


# Helper classes for computing numerical collision detection using PyTorch.
# These mirror the symbolic versions but use torch tensors for autodiff.


class NumericRegionHalfspace:
    """Numeric collision detection for RegionHalfspace using PyTorch."""

    def __init__(self, region: RegionHalfspace):
        self.normal = torch.tensor(region.normal, dtype=torch.float64)
        self.anchor = torch.tensor(region.anchor, dtype=torch.float64)

    def point(self, coordinate: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Compute the residual error of the point on this halfspace.

        Positive return values indicate that the point lies inside the halfspace.
        """
        x, y = coordinate
        # Vector from anchor to point
        dx = x - self.anchor[0]
        dy = y - self.anchor[1]

        # Dot product with normal: normal · (point - anchor)
        return self.normal[0] * dx + self.normal[1] * dy

    def line(
        self,
        c1: Tuple[torch.Tensor, torch.Tensor],
        c2: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the residual error of the line from c1 to c2 on this halfspace."""
        return torch.clamp(self.point(c1), min=0.0) + torch.clamp(
            self.point(c2), min=0.0
        )


class NumericRegionBall:
    """Numeric collision detection for RegionBall using PyTorch."""

    def __init__(self, region: RegionBall):
        self.center = torch.tensor(region.center, dtype=torch.float64)
        self.radius = region.radius

    def point(self, coordinate: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Compute the residual error of the point on this ball.

        Positive return values indicate that the point lies inside the ball.
        """
        x, y = coordinate
        # Distance from center to point
        dx = x - self.center[0]
        dy = y - self.center[1]
        distance = torch.sqrt(dx**2 + dy**2)

        # Return radius - distance (positive when inside)
        return self.radius - distance

    def line(
        self,
        c1: Tuple[torch.Tensor, torch.Tensor],
        c2: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the residual error of the line segment from c1 to c2 on this ball."""
        # First check if either endpoint is inside the ball
        p1 = self.point(c1)
        p2 = self.point(c2)
        endpoint_collision = torch.clamp(p1, min=0.0) + torch.clamp(p2, min=0.0)

        # Compute projection of ball center onto the line segment
        dx = c2[0] - c1[0]
        dy = c2[1] - c1[1]

        # Vector from c1 to ball center
        cx = self.center[0] - c1[0]
        cy = self.center[1] - c1[1]

        # Compute length squared of line segment
        length_sq = dx**2 + dy**2

        # Compute projection parameter t
        t = (cx * dx + cy * dy) / (length_sq + 1e-10)

        # Clamp t to [0, 1]
        t_clamped = torch.clamp(t, min=0.0, max=1.0)

        # Compute the closest point on the line segment
        closest_x = c1[0] + t_clamped * dx
        closest_y = c1[1] + t_clamped * dy

        # Compute distance from ball center to closest point
        dist_x = self.center[0] - closest_x
        dist_y = self.center[1] - closest_y
        perpendicular_distance = torch.sqrt(dist_x**2 + dist_y**2)

        # Collision if perpendicular distance < radius
        segment_collision = torch.clamp(self.radius - perpendicular_distance, min=0.0)

        return torch.maximum(endpoint_collision, segment_collision)


class NumericRegionRectangle:
    """Numeric collision detection for RegionRectangle using PyTorch."""

    def __init__(self, region: RegionRectangle):
        self.left = region.left
        self.right = region.right
        self.bottom = region.bottom
        self.top = region.top

    def point(self, coordinate: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Compute the residual error to the closest boundary.

        Positive return values indicate that the point lies inside the rectangle.
        """
        x, y = coordinate

        # Distance to each boundary (positive when inside)
        dist_to_left = x - self.left
        dist_to_right = self.right - x
        dist_to_bottom = y - self.bottom
        dist_to_top = self.top - y

        # Minimum distance determines the residual
        return torch.minimum(
            torch.minimum(dist_to_left, dist_to_right),
            torch.minimum(dist_to_bottom, dist_to_top),
        )

    def line(
        self,
        c1: Tuple[torch.Tensor, torch.Tensor],
        c2: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the residual error of the line segment from c1 to c2 on this rectangle."""
        # Check endpoint collisions
        p1 = self.point(c1)
        p2 = self.point(c2)
        endpoint_collision = torch.clamp(p1, min=0.0) + torch.clamp(p2, min=0.0)

        x1, y1 = c1
        x2, y2 = c2

        # Vector from c1 to c2
        dx = x2 - x1
        dy = y2 - y1

        def edge_intersection_score(
            t: torch.Tensor, edge_coord: torch.Tensor, edge_min: float, edge_max: float
        ) -> torch.Tensor:
            # Check t in [0, 1]
            t_valid = torch.minimum(t, 1.0 - t)
            # Check coord in [edge_min, edge_max]
            coord_valid = torch.minimum(edge_coord - edge_min, edge_max - edge_coord)
            # Both must be non-negative
            return torch.clamp(torch.minimum(t_valid, coord_valid), min=0.0)

        # Left edge
        t_left = (self.left - x1) / (dx + 1e-10)
        y_at_left = y1 + t_left * dy
        left_collision = edge_intersection_score(
            t_left, y_at_left, self.bottom, self.top
        )

        # Right edge
        t_right = (self.right - x1) / (dx + 1e-10)
        y_at_right = y1 + t_right * dy
        right_collision = edge_intersection_score(
            t_right, y_at_right, self.bottom, self.top
        )

        # Bottom edge
        t_bottom = (self.bottom - y1) / (dy + 1e-10)
        x_at_bottom = x1 + t_bottom * dx
        bottom_collision = edge_intersection_score(
            t_bottom, x_at_bottom, self.left, self.right
        )

        # Top edge
        t_top = (self.top - y1) / (dy + 1e-10)
        x_at_top = x1 + t_top * dx
        top_collision = edge_intersection_score(t_top, x_at_top, self.left, self.right)

        edge_collision = (
            left_collision + right_collision + bottom_collision + top_collision
        )

        return torch.maximum(endpoint_collision, edge_collision)


def make_numeric_region(region: Region):
    """Factory function to create the appropriate numeric region helper."""
    if isinstance(region, RegionHalfspace):
        return NumericRegionHalfspace(region)
    elif isinstance(region, RegionBall):
        return NumericRegionBall(region)
    elif isinstance(region, RegionRectangle):
        return NumericRegionRectangle(region)
    else:
        raise TypeError(f"Unknown region type: {type(region)}")


@dataclass
class IKNumericProfile:
    """Profiling results from IKNumeric solver."""

    solve_time_ms: float  # Total solve time in milliseconds
    iterations: int  # Number of optimization iterations
    converged: bool  # Whether the solver converged before max_iterations
    initial_loss: float  # Loss at the start of optimization
    final_loss: float  # Loss at the end of optimization
    position_error: float  # Final Euclidean distance to target position


class IKNumeric:
    """Implements a numerical solver for inverse kinematics using PyTorch autodiff."""

    def __init__(
        self,
        model: RobotModel,
        world: WorldModel | None,
        lr: float = 0.02,
        max_iterations: int = 500,
        tolerance: float = 1e-6,
        collision_geometry: Literal["line", "point"] = "line",
    ) -> None:
        self.model = model
        self.nogo = world and world.nogo
        self.n_joints = len(model.link_lengths)
        self.collision_geometry = collision_geometry

        # Optimization parameters
        self.lr = lr
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Convert model parameters to tensors
        self.link_lengths = torch.tensor(model.link_lengths, dtype=torch.float64)
        self.joint_origins = torch.tensor(model.joint_origins, dtype=torch.float64)

        # Pre-build numeric region helpers if nogo zones exist
        self.numeric_regions = None
        if self.nogo:
            self.numeric_regions = [make_numeric_region(r) for r in self.nogo]

    def _forward_kinematics(
        self, thetas: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """Compute forward kinematics using PyTorch tensors.

        Returns:
            Tuple of (end_effector_x, end_effector_y, end_effector_angle, joint_positions)
            where joint_positions is a list of (x, y) tuples for each joint including origin.
        """
        x = torch.tensor(0.0, dtype=torch.float64)
        y = torch.tensor(0.0, dtype=torch.float64)
        cumulative_angle = torch.tensor(0.0, dtype=torch.float64)

        # Store joint positions including origin
        joint_positions = [(x.clone(), y.clone())]

        for i in range(self.n_joints):
            cumulative_angle = cumulative_angle + thetas[i] + self.joint_origins[i]
            x = x + self.link_lengths[i] * torch.cos(cumulative_angle)
            y = y + self.link_lengths[i] * torch.sin(cumulative_angle)
            joint_positions.append((x.clone(), y.clone()))

        return x, y, cumulative_angle, joint_positions

    def _compute_objective(
        self,
        thetas: torch.Tensor,
        target_x: float,
        target_y: float,
        target_angle: float,
        angle_weight: float,
        nogo_weight: float,
    ) -> torch.Tensor:
        """Compute the objective function value."""
        ee_x, ee_y, ee_angle, joint_positions = self._forward_kinematics(thetas)

        # Position error
        distance_squared = (ee_x - target_x) ** 2 + (ee_y - target_y) ** 2

        # Angle error with wrapping
        angle_diff = ee_angle - target_angle
        angle_error = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        angle_error_squared = angle_error**2

        objective = distance_squared + angle_weight * angle_error_squared

        # Add nogo zone penalty
        if self.numeric_regions:
            nogo_penalty = torch.tensor(0.0, dtype=torch.float64)

            if self.collision_geometry == "line":
                # For each link segment
                for i in range(len(joint_positions) - 1):
                    p1 = joint_positions[i]
                    p2 = joint_positions[i + 1]

                    # Check each nogo region
                    for numeric_region in self.numeric_regions:
                        segment_penalty = numeric_region.line(p1, p2)
                        nogo_penalty = nogo_penalty + segment_penalty**2
            else:  # point
                # Only check joint positions (excluding origin at index 0)
                for i in range(1, len(joint_positions)):
                    p = joint_positions[i]

                    # Check each nogo region
                    for numeric_region in self.numeric_regions:
                        point_penalty = numeric_region.point(p)
                        # Clamp to only penalize when inside the region (positive values)
                        point_penalty = torch.clamp(point_penalty, min=0.0)
                        nogo_penalty = nogo_penalty + point_penalty**2

            objective = objective + nogo_weight * nogo_penalty

        return objective

    def __call__(
        self,
        state: RobotState,
        desired: DesiredPosition,
        profile: bool = False,
    ) -> RobotState | Tuple[RobotState, IKNumericProfile]:
        """Solve IK for the desired position.

        Args:
            state: Current robot state.
            desired: Desired end effector position and optional angle.
            profile: If True, return profiling information along with the result.

        Returns:
            If profile is False: RobotState with the solution.
            If profile is True: Tuple of (RobotState, IKNumericProfile).
        """
        if state.model != self.model:
            raise ValueError("State model does not match IKNumeric model")

        if desired.ee_position is None:
            raise ValueError("DesiredPosition must have an ee_position")

        target_x, target_y = desired.ee_position

        # Extract angle constraint
        if desired.ee_angle is not None:
            target_angle = desired.ee_angle
            angle_weight = 1.0e3
        else:
            target_angle = 0.0
            angle_weight = 0.0

        # Nogo weight
        nogo_weight = 1.0e2 if self.numeric_regions else 0.0

        # Initialize joint angles from current state
        thetas = torch.tensor(
            list(state.current.joint_angles), dtype=torch.float64, requires_grad=True
        )

        # Use Adam optimizer with learning rate annealing
        optimizer = torch.optim.Adam([thetas], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        # Start profiling
        start_time = time.perf_counter()

        prev_loss = float("inf")
        current_loss = float("inf")
        initial_loss: Optional[float] = None
        converged = False
        iterations_completed = 0

        for iteration in range(self.max_iterations):
            optimizer.zero_grad()

            loss = self._compute_objective(
                thetas, target_x, target_y, target_angle, angle_weight, nogo_weight
            )

            # Record initial loss
            if initial_loss is None:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

            iterations_completed = iteration + 1

            # Check convergence
            current_loss = loss.item()
            if abs(prev_loss - current_loss) < self.tolerance:
                converged = True
                break
            prev_loss = current_loss

        # End profiling
        end_time = time.perf_counter()
        final_loss = current_loss

        # Extract solution
        joint_angles = tuple(float(angle) for angle in thetas.detach().numpy())

        result_state = state.with_position(
            RobotPosition(joint_angles=joint_angles), desired=desired
        )

        if profile:
            # Compute final position error
            with torch.no_grad():
                ee_x, ee_y, _, _ = self._forward_kinematics(thetas)
                position_error = float(
                    torch.sqrt((ee_x - target_x) ** 2 + (ee_y - target_y) ** 2)
                )

            profile_result = IKNumericProfile(
                solve_time_ms=(end_time - start_time) * 1000,
                iterations=iterations_completed,
                converged=converged,
                initial_loss=initial_loss if initial_loss is not None else 0.0,
                final_loss=final_loss,
                position_error=position_error,
            )
            return result_state, profile_result

        return result_state


if __name__ == "__main__":
    # Interactive IK solver demo using RobotVisualizer
    import math

    from visualization import RobotVisualizer

    # Create a 3-link robot
    model = RobotModel(link_lengths=(1.0, 1.0, 0.6))
    # Create a world with a narrow space to enter.
    nogo = [
        RegionHalfspace((0, -1), (0, -0.2)),
        RegionRectangle(0.5, 10.0, -10.0, 1.0),
        RegionRectangle(0.5, 10.0, 1.6, 5.0),
    ]
    world = WorldModel(nogo=nogo)

    # Create the IK solver
    ik_solver = IKNumeric(
        model, world=world, collision_geometry="line", max_iterations=200, lr=0.06
    )

    # Initial position
    initial_position = RobotPosition(
        joint_angles=(2.5 * math.pi / 4, -math.pi + 0.1, math.pi - 0.1)
    )
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
            print(f"Loss: {profile.initial_loss:.6f} -> {profile.final_loss:.6f}")
            print(f"Position error: {profile.position_error:.6f}")

            # Update the visualization with the new solution
            current_state = solution_state
            viz.update(current_state)

        except Exception as e:
            print(f"Error solving IK: {e}")

    # Set the click callback
    viz.set_click_callback(on_click)

    print("Interactive IK Solver (Numeric/PyTorch)")
    print("=" * 60)
    print("Click anywhere in the window to set a target position.")
    print("The robot will solve IK and move to reach that target.")
    print("=" * 60)
    print(f"Callback registered: {viz.click_callback is not None}")
    print(f"Event handler connected: {hasattr(viz, '_click_handler_id')}")

    # Show the visualization
    viz.show()
