#!python3

import sympy as sp
from typing import Any, Callable, Tuple

from datamodel import Position, RobotModel, RobotPosition, RobotState


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

        # Create distance squared function
        distance_squared = (x_sym - target_x) ** 2 + (y_sym - target_y) ** 2

        # Simplify the distance function
        distance_squared_simplified = sp.simplify(distance_squared)

        # Compute the gradient (derivative with respect to each joint angle)
        gradient = [
            sp.diff(distance_squared_simplified, theta) for theta in self.theta_symbols
        ]
        gradient_simplified = [sp.simplify(g) for g in gradient]

        # Convert to numerical functions
        # Distance function takes (target_x, target_y, theta0, theta1, ...)
        self.distance_func = sp.lambdify(
            [target_x, target_y] + list(self.theta_symbols),
            distance_squared_simplified,
            "numpy",
        )

        # Gradient function takes (target_x, target_y, theta0, theta1, ...)
        self.gradient_func = sp.lambdify(
            [target_x, target_y] + list(self.theta_symbols),
            gradient_simplified,
            "numpy",
        )

    def __call__(self, state: RobotState) -> RobotPosition:
        # Sanity-check that state.model is the same as self.model
        if state.model != self.model:
            raise ValueError("State model does not match IKSymbolic model")

        # Get the desired end effector position
        if state.desired_end_effector is None:
            raise ValueError("State must have a desired_end_effector position")

        target_x, target_y = state.desired_end_effector

        # Use scipy for optimization with precomputed distance and gradient functions
        from scipy.optimize import minimize

        # Initial guess from current state
        x0 = list(state.current.joint_angles)

        # Define objective function and gradient with fixed target
        def objective(thetas):
            return self.distance_func(target_x, target_y, *thetas)

        def gradient(thetas):
            return self.gradient_func(target_x, target_y, *thetas)

        # Minimize distance to target using BFGS with analytical gradient
        result = minimize(objective, x0, method="BFGS", jac=gradient)

        # Extract joint angles from solution
        joint_angles = tuple(float(angle) for angle in result.x)

        return RobotPosition(joint_angles=joint_angles)


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
    current_state = RobotState(model, initial_position, None)

    # Create visualizer
    viz = RobotVisualizer(current_state)

    # Click callback that updates the target and solves IK
    def on_click(x: float, y: float):
        global current_state
        print(f"\nClicked at: ({x:.2f}, {y:.2f})")

        # Update the desired end effector position
        new_state = RobotState(model, current_state.current, (x, y))

        # Solve IK
        try:
            solution = ik_solver(new_state)
            print(
                f"Solution joint angles: {tuple(f'{a:.3f}' for a in solution.joint_angles)}"
            )

            # Verify the solution
            end_effector_positions = model.forward_kinematics(solution)
            end_effector = end_effector_positions[-1]
            error = math.sqrt((end_effector[0] - x) ** 2 + (end_effector[1] - y) ** 2)
            print(f"Achieved position: ({end_effector[0]:.3f}, {end_effector[1]:.3f})")
            print(f"Position error: {error:.6f}")

            # Update the visualization with the new solution
            current_state = RobotState(model, solution, (x, y))
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
