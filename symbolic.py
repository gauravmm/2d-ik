#!python3

from typing import Literal, Optional, Tuple

import sympy as sp
from scipy.optimize import minimize

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


# Helper classes for computing symbolic expressions for region collision detection.
# These are separated from the Region dataclasses to allow different solvers to use
# different implementations (e.g., symbolic vs numerical).


class SymbolicRegionHalfspace:
    """Symbolic collision detection for RegionHalfspace."""

    def __init__(self, region: RegionHalfspace):
        self.region = region

    def point(self, coordinate: Tuple[sp.Expr, sp.Expr]) -> sp.Expr:
        """Compute the residual error of the point on this halfspace.

        Positive return values indicate that the point lies inside the halfspace.

        Returns:
            Signed distance: normal · (point - anchor)
            Positive = inside, negative = outside
        """
        x, y = coordinate
        # Vector from anchor to point
        dx = x - self.region.anchor[0]
        dy = y - self.region.anchor[1]

        # Dot product with normal: normal · (point - anchor)
        return self.region.normal[0] * dx + self.region.normal[1] * dy

    def line(self, c1: Tuple[sp.Expr, sp.Expr], c2: Tuple[sp.Expr, sp.Expr]) -> sp.Expr:
        """Compute the residual error of the line from c1 to c2 on this halfspace.

        Positive return values indicate that some segment of the line lies inside the
        halfspace. The return value does not have any specific interpretable meaning.

        Returns: residual error, if any
        """
        return sp.Max(self.point(c1), 0.0) + sp.Max(self.point(c2), 0.0)


class SymbolicRegionBall:
    """Symbolic collision detection for RegionBall."""

    def __init__(self, region: RegionBall):
        self.region = region

    def point(self, coordinate: Tuple[sp.Expr, sp.Expr]) -> sp.Expr:
        """Compute the residual error of the point on this ball.

        Positive return values indicate that the point lies inside the ball.

        Returns:
            radius - distance_to_center
            Positive = inside, negative = outside
        """
        x, y = coordinate
        # Distance from center to point
        dx = x - self.region.center[0]
        dy = y - self.region.center[1]
        distance = sp.sqrt(dx**2 + dy**2)

        # Return radius - distance (positive when inside)
        return self.region.radius - distance

    def line(self, c1: Tuple[sp.Expr, sp.Expr], c2: Tuple[sp.Expr, sp.Expr]) -> sp.Expr:
        """Compute the residual error of the line segment from c1 to c2 on this ball.
        Positive return values indicate that the line segment collides with the ball.

        Returns: residual error, positive indicates collision.
        """
        # First check if either endpoint is inside the ball
        p1 = self.point(c1)
        p2 = self.point(c2)
        endpoint_collision = sp.Max(p1, 0.0) + sp.Max(p2, 0.0)

        # Compute projection of ball center onto the line segment
        # Vector from c1 to c2
        dx = c2[0] - c1[0]
        dy = c2[1] - c1[1]

        # Vector from c1 to ball center
        cx = self.region.center[0] - c1[0]
        cy = self.region.center[1] - c1[1]

        # Compute length squared of line segment
        length_sq = dx**2 + dy**2

        # Compute projection parameter t: dot(c1->center, c1->c2) / ||c1->c2||^2
        # t = 0 means projection is at c1, t = 1 means projection is at c2
        t = (cx * dx + cy * dy) / (length_sq + 1e-10)

        # Clamp t to [0, 1] to get the closest point on the line segment
        t_clamped = sp.Max(0.0, sp.Min(1.0, t))

        # Compute the closest point on the line segment
        closest_x = c1[0] + t_clamped * dx
        closest_y = c1[1] + t_clamped * dy

        # Compute distance from ball center to closest point on line segment
        dist_x = self.region.center[0] - closest_x
        dist_y = self.region.center[1] - closest_y
        perpendicular_distance = sp.sqrt(dist_x**2 + dist_y**2)

        # Collision occurs if perpendicular distance is less than radius
        # Return radius - perpendicular_distance (positive if collision)
        segment_collision = sp.Max(self.region.radius - perpendicular_distance, 0.0)

        # Return the maximum of endpoint collision and segment collision
        return sp.Max(endpoint_collision, segment_collision)


class SymbolicRegionRectangle:
    """Symbolic collision detection for RegionRectangle."""

    def __init__(self, region: RegionRectangle):
        self.region = region

    def point(self, coordinate: Tuple[sp.Expr, sp.Expr]) -> sp.Expr:
        """Compute the residual error to the closest boundary.

        Positive return values indicate that the point lies inside the rectangle.

        Returns:
            Minimum distance to any boundary (negative if outside)
            Positive = inside, negative = outside

        For points inside: returns the distance to the nearest edge
        For points outside: returns the negative distance to the nearest edge
        """
        x, y = coordinate

        # Distance to each boundary (positive when inside)
        dist_to_left = x - self.region.left  # Positive when x > left
        dist_to_right = self.region.right - x  # Positive when x < right
        dist_to_bottom = y - self.region.bottom  # Positive when y > bottom
        dist_to_top = self.region.top - y  # Positive when y < top

        # The minimum of these distances determines the residual
        # If all are positive, point is inside and residual is distance to nearest edge
        # If any is negative, point is outside and residual is the most negative value
        # Using sympy's Min to handle symbolic expressions
        return sp.Min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)

    def line(self, c1: Tuple[sp.Expr, sp.Expr], c2: Tuple[sp.Expr, sp.Expr]) -> sp.Expr:
        """Compute the residual error of the line segment from c1 to c2 on this rectangle.

        Positive return values indicate that the line segment collides with the rectangle.
        Uses exact geometric intersection: returns positive only if the line segment
        actually intersects the rectangle.

        Returns: residual error, positive indicates collision.
        """
        # Check endpoint collisions
        p1 = self.point(c1)
        p2 = self.point(c2)
        endpoint_collision = sp.Max(p1, 0.0) + sp.Max(p2, 0.0)

        x1, y1 = c1
        x2, y2 = c2

        # Vector from c1 to c2
        dx = x2 - x1
        dy = y2 - y1

        # Check intersection with each of the 4 rectangle edges
        # For each edge, compute parameter t where segment intersects edge
        # Valid intersection requires t in [0, 1] and intersection point on edge
        # We use a smooth approximation: if conditions are met, return positive value

        # Helper function: returns positive if all conditions met, using smooth max/min
        # If t in [0,1] and point in edge bounds, return 1.0, else 0.0
        def edge_intersection_score(
            t: sp.Expr, edge_coord: sp.Expr, edge_min: float, edge_max: float
        ) -> sp.Expr:
            # Check t in [0, 1]: min(t, 1-t) >= 0
            t_valid = sp.Min(t, 1.0 - t)
            # Check coord in [edge_min, edge_max]: min(coord - edge_min, edge_max - coord) >= 0
            coord_valid = sp.Min(edge_coord - edge_min, edge_max - edge_coord)
            # Both must be non-negative for valid intersection
            # Return positive value if both are >= 0
            return sp.Max(0.0, sp.Min(t_valid, coord_valid))

        # Left edge (x = left, y in [bottom, top])
        t_left = (self.region.left - x1) / (dx + 1e-10)
        y_at_left = y1 + t_left * dy
        left_collision = edge_intersection_score(
            t_left, y_at_left, self.region.bottom, self.region.top
        )

        # Right edge (x = right, y in [bottom, top])
        t_right = (self.region.right - x1) / (dx + 1e-10)
        y_at_right = y1 + t_right * dy
        right_collision = edge_intersection_score(
            t_right, y_at_right, self.region.bottom, self.region.top
        )

        # Bottom edge (y = bottom, x in [left, right])
        t_bottom = (self.region.bottom - y1) / (dy + 1e-10)
        x_at_bottom = x1 + t_bottom * dx
        bottom_collision = edge_intersection_score(
            t_bottom, x_at_bottom, self.region.left, self.region.right
        )

        # Top edge (y = top, x in [left, right])
        t_top = (self.region.top - y1) / (dy + 1e-10)
        x_at_top = x1 + t_top * dx
        top_collision = edge_intersection_score(
            t_top, x_at_top, self.region.left, self.region.right
        )

        # Total edge collision is the sum of all edge intersections
        edge_collision = (
            left_collision + right_collision + bottom_collision + top_collision
        )

        # Return maximum of endpoint and edge collisions
        return sp.Max(endpoint_collision, edge_collision)


def make_symbolic_region(region: Region):
    """Factory function to create the appropriate symbolic region helper."""
    if isinstance(region, RegionHalfspace):
        return SymbolicRegionHalfspace(region)
    elif isinstance(region, RegionBall):
        return SymbolicRegionBall(region)
    elif isinstance(region, RegionRectangle):
        return SymbolicRegionRectangle(region)
    else:
        raise TypeError(f"Unknown region type: {type(region)}")


class IKSymbolic:
    """Implements a symbolic solver for inverse kinematics using the Sympy solver."""

    def __init__(
        self,
        model: RobotModel,
        world: WorldModel | None,
        collision_geometry: Literal["line", "point"] = "line",
    ) -> None:
        # Create variables for each joint rotation and set up a system of equations that
        # relate each end effector position
        self.model = model
        self.nogo = world and world.nogo
        self.n_joints = len(model.link_lengths)
        self.collision_geometry = collision_geometry

        # Create symbolic variables for joint angles
        thetas = sp.symbols(f"theta0:{self.n_joints}", real=True, seq=True)
        assert isinstance(thetas, tuple)
        self.theta_symbols: Tuple[sp.Symbol, ...] = thetas

        # Create symbolic variables for target position
        target_x = sp.Symbol("target_x", real=True)
        target_y = sp.Symbol("target_y", real=True)
        target_angle = sp.Symbol("target_angle", real=True)
        angle_weight = sp.Symbol("angle_weight", real=True, positive=True)

        # Build symbolic forward kinematics equations
        # Store positions and angles at each joint for boundary checking
        joint_x_syms: list[sp.Expr] = []
        joint_y_syms: list[sp.Expr] = []
        joint_angles_syms: list[sp.Expr] = []

        x_sym: sp.Expr = sp.Float(0)
        y_sym: sp.Expr = sp.Float(0)
        cumulative_angle = sp.Float(0)

        for i, (link_length, joint_origin) in enumerate(
            zip(model.link_lengths, model.joint_origins)
        ):
            # Add joint angle and origin to cumulative angle
            cumulative_angle = cumulative_angle + self.theta_symbols[i] + joint_origin

            # Add this link's contribution to end effector position
            x_sym = x_sym + link_length * sp.cos(cumulative_angle)
            y_sym = y_sym + link_length * sp.sin(cumulative_angle)

            # Store position and angle at this joint
            joint_x_syms.append(x_sym)
            joint_y_syms.append(y_sym)
            joint_angles_syms.append(cumulative_angle)

        # Store the symbolic end effector position equations
        self.end_effector_x: sp.Expr = x_sym
        self.end_effector_y: sp.Expr = y_sym

        # Store the symbolic end effector orientation
        self.end_effector_angle: sp.Expr = cumulative_angle

        # Store symbolic positions and angles at each joint for boundary checking
        self.joint_x_syms: Tuple[sp.Expr, ...] = tuple(joint_x_syms)
        self.joint_y_syms: Tuple[sp.Expr, ...] = tuple(joint_y_syms)
        self.joint_angles_syms: Tuple[sp.Expr, ...] = tuple(joint_angles_syms)

        # Create distance squared function
        distance_squared = (x_sym - target_x) ** 2 + (y_sym - target_y) ** 2

        # Create angle error with wrapping (shortest angular distance)
        # Using atan2(sin(diff), cos(diff)) ensures smooth wrapping to [-π, π]
        angle_diff = self.end_effector_angle - target_angle
        angle_error = sp.atan2(sp.sin(angle_diff), sp.cos(angle_diff))
        angle_error_squared = angle_error**2

        # Combined objective: position error + weighted angle error
        # When angle_weight = 0, this reduces to position-only optimization
        combined_objective = distance_squared + angle_weight * angle_error_squared
        # Simplify the combined objective function
        combined_objective = sp.simplify(combined_objective)

        # Add nogo zone penalty for each link
        if self.nogo:
            print(
                "ERROR: nogo zones are not supported by the IKSymbolic solver due to slowdown in the simplify steps."
            )
            raise NotImplementedError()
            nogo_weight = sp.Symbol("nogo_weight", real=True, positive=True)
            nogo_penalty: sp.Expr = sp.Float(0)

            # Build list of joint positions including origin
            joint_positions: list[tuple[sp.Expr, sp.Expr]] = [
                (sp.Float(0), sp.Float(0))
            ]
            for jx, jy in zip(joint_x_syms, joint_y_syms):
                joint_positions.append((jx, jy))

            if self.collision_geometry == "line":
                # For each link (segment between consecutive joints)
                for i in range(len(joint_positions) - 1):
                    p1 = joint_positions[i]
                    p2 = joint_positions[i + 1]

                    # Check each nogo region
                    for region in self.nogo:
                        # Use symbolic region helper to get penalty for this segment
                        symbolic_region = make_symbolic_region(region)
                        segment_penalty = symbolic_region.line(p1, p2)
                        # Square the penalty to make it smooth and differentiable
                        nogo_penalty = nogo_penalty + segment_penalty**2
            else:  # point
                # Only check joint positions (excluding origin at index 0)
                for i in range(1, len(joint_positions)):
                    p = joint_positions[i]

                    # Check each nogo region
                    for region in self.nogo:
                        symbolic_region = make_symbolic_region(region)
                        point_penalty = symbolic_region.point(p)
                        # Use Max to only penalize when inside the region (positive values)
                        point_penalty = sp.Max(point_penalty, 0.0)
                        # Square the penalty to make it smooth and differentiable
                        nogo_penalty = nogo_penalty + point_penalty**2

            combined_objective = combined_objective + nogo_weight * nogo_penalty
            self.nogo_weight_symbol = nogo_weight
        else:
            self.nogo_weight_symbol = None

        # Compute the gradient (derivative with respect to each joint angle)
        gradient = [sp.diff(combined_objective, theta) for theta in self.theta_symbols]
        # gradient_simplified = [sp.simplify(g) for g in gradient]

        # Convert to numerical functions
        # Build parameter list: (target_x, target_y, target_angle, angle_weight, [nogo_weight], theta0, theta1, ...)
        base_params = [target_x, target_y, target_angle, angle_weight]
        if self.nogo_weight_symbol is not None:
            base_params.append(self.nogo_weight_symbol)

        self.combined_objective_func = sp.lambdify(
            base_params + list(self.theta_symbols),
            combined_objective,
            "numpy",
        )

        self.combined_gradient_func = sp.lambdify(
            base_params + list(self.theta_symbols),
            gradient,
            "numpy",
        )

        # Function to compute end effector angle for verification, primarily for testing.
        self.angle_func = sp.lambdify(
            list(self.theta_symbols),
            self.end_effector_angle,
            "numpy",
        )

    def __call__(
        self,
        state: RobotState,
        desired: DesiredPosition,
    ) -> RobotState:
        # Sanity-check that state.model is the same as self.model
        if state.model != self.model:
            raise ValueError("State model does not match IKSymbolic model")

        # Known bug:
        if state.world.nogo:
            print(
                "ERROR: nogo zones are not supported by the IKSymbolic solver due to slowdown in the simplify steps."
            )

        # Get the desired end effector position
        if desired.ee_position is None:
            raise ValueError("DesiredPosition must have an ee_position")

        target_x, target_y = desired.ee_position

        # Extract angle constraint if present
        if desired.ee_angle is not None:
            target_angle = desired.ee_angle
            # Default angle weight - could be made configurable via RobotModel
            angle_weight = 1.0e3
        else:
            # No angle constraint - use dummy values with zero weight
            target_angle = 0.0  # Dummy value, won't affect optimization
            angle_weight = 0.0  # Zero weight disables angle constraint

        # Initial guess from current state
        x0 = list(state.current.joint_angles)

        # Build base parameters for objective/gradient calls
        base_args = [target_x, target_y, target_angle, angle_weight]
        if self.nogo_weight_symbol is not None:
            # Default nogo weight - high penalty for violations
            nogo_weight = 1.0e4
            base_args.append(nogo_weight)

        # Define objective function and gradient with fixed target and weight
        def objective(thetas):
            return self.combined_objective_func(*base_args, *thetas)

        def gradient(thetas):
            grad_result = self.combined_gradient_func(*base_args, *thetas)
            # Handle case where gradient might be a single value (1 joint) or array
            if hasattr(grad_result, "__iter__"):
                return grad_result
            else:
                return [grad_result]

        # Minimize distance to target using BFGS with analytical gradient
        result = minimize(objective, x0, method="BFGS", jac=gradient)

        # Extract joint angles from solution
        joint_angles = tuple(float(angle) for angle in result.x)

        return state.with_position(
            RobotPosition(joint_angles=joint_angles), desired=desired
        )


if __name__ == "__main__":
    # Interactive IK solver demo using RobotVisualizer
    import math

    from visualization import RobotVisualizer

    # Create a 3-link robot
    model = RobotModel(link_lengths=(1.0, 0.8, 0.6))
    # Create a world with a narrow space to enter.
    nogo = [
        RegionHalfspace((0, -1), (0, -0.2)),
        RegionRectangle(0.5, 10.0, -10.0, 1.0),
        RegionRectangle(0.5, 10.0, 1.6, 5.0),
    ]
    world = WorldModel(nogo=None)

    # Create the IK solver
    ik_solver = IKSymbolic(model, world=world, collision_geometry="point")

    # Initial position
    initial_position = RobotPosition(
        joint_angles=(2.5 * math.pi / 4, -math.pi + 0.1, math.pi - 0.1)
    )
    global current_state
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

        # Solve IK
        try:
            solution_state = ik_solver(
                current_state,
                DesiredPosition(ee_position=(x, y), ee_angle=new_ee_angle),
            )
            solution = solution_state.current
            print(f"Solution: {tuple(f'{a:.3f}' for a in solution.joint_angles)}")

            # Verify the solution
            end_effector_positions = model.forward_kinematics(solution)
            end_effector = end_effector_positions[-1]
            error = math.sqrt((end_effector[0] - x) ** 2 + (end_effector[1] - y) ** 2)
            print(f"Position error: {error:.6f}")

            # Update the visualization with the new solution
            current_state = solution_state
            viz.update(current_state)

        except Exception as e:
            print(f"Error solving IK: {e}")

    # Set the click callback
    viz.set_click_callback(on_click)

    print("Interactive IK Solver")
    print("=" * 60)
    print("Click anywhere in the window to set a target position.")
    print("The robot will solve IK and move to reach that target.")
    print("=" * 60)
    print(f"Callback registered: {viz.click_callback is not None}")
    print(f"Event handler connected: {hasattr(viz, '_click_handler_id')}")

    # Show the visualization
    viz.show()
