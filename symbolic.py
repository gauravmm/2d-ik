#!python3

from typing import Literal, Optional, Tuple

import sympy as sp
from scipy.optimize import minimize

from datamodel import DesiredPosition, RobotModel, RobotPosition, RobotState


class IKSymbolic:
    """Implements a symbolic solver for inverse kinematics using the Sympy solver."""

    def __init__(self, model: RobotModel) -> None:
        # Create variables for each joint rotation and set up a system of equations that
        # relate each end effector position
        self.model = model
        self.n_joints = len(model.link_lengths)

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

        # Store the symbolic end effector position equations
        self.end_effector_x: sp.Expr = x_sym
        self.end_effector_y: sp.Expr = y_sym

        # Store the symbolic end effector orientation
        self.end_effector_angle: sp.Expr = cumulative_angle

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
        combined_objective_simplified = sp.simplify(combined_objective)

        # Compute the gradient (derivative with respect to each joint angle)
        gradient = [
            sp.diff(combined_objective_simplified, theta)
            for theta in self.theta_symbols
        ]
        gradient_simplified = [sp.simplify(g) for g in gradient]

        # Convert to numerical functions
        # Objective function takes (target_x, target_y, target_angle, angle_weight, theta0, theta1, ...)
        self.combined_objective_func = sp.lambdify(
            [target_x, target_y, target_angle, angle_weight] + list(self.theta_symbols),
            combined_objective_simplified,
            "numpy",
        )

        # Gradient function takes (target_x, target_y, target_angle, angle_weight, theta0, theta1, ...)
        self.combined_gradient_func = sp.lambdify(
            [target_x, target_y, target_angle, angle_weight] + list(self.theta_symbols),
            gradient_simplified,
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

        # Define objective function and gradient with fixed target and weight
        def objective(thetas):
            return self.combined_objective_func(
                target_x, target_y, target_angle, angle_weight, *thetas
            )

        def gradient(thetas):
            grad_result = self.combined_gradient_func(
                target_x, target_y, target_angle, angle_weight, *thetas
            )
            # Handle case where gradient might be a single value (1 joint) or array
            if hasattr(grad_result, "__iter__"):
                return grad_result
            else:
                return [grad_result]

        # Minimize distance to target using BFGS with analytical gradient
        result = minimize(objective, x0, method="BFGS", jac=gradient)

        # Extract joint angles from solution
        joint_angles = tuple(float(angle) for angle in result.x)

        return RobotState(
            state.model,
            RobotPosition(joint_angles=joint_angles),
            desired=desired,
        )


if __name__ == "__main__":
    # Interactive IK solver demo using RobotVisualizer
    import math

    from visualization import RobotVisualizer

    # Create a 3-link robot
    model = RobotModel(link_lengths=(1.0, 0.8, 0.6))

    # Create the IK solver
    ik_solver = IKSymbolic(model)

    # Initial position
    initial_position = RobotPosition(joint_angles=(0.0, math.pi / 4, -math.pi / 4))
    global current_state
    current_state = RobotState(model, initial_position)

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

        # Create input state with current position
        input_state = RobotState(model, current_state.current)

        # Solve IK
        try:
            solution_state = ik_solver(
                input_state, DesiredPosition(ee_position=(x, y), ee_angle=new_ee_angle)
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
