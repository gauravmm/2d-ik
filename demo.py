#!/usr/bin/env python3
"""Interactive IK solver demo with support for multiple solver backends."""

import argparse
import math
from typing import Literal, Optional

from datamodel import (
    DesiredPosition,
    RegionHalfspace,
    RegionRectangle,
    RobotModel,
    RobotPosition,
    RobotState,
    WorldModel,
)
from visualization import RobotVisualizer


def create_solver(
    solver_type: str,
    model: RobotModel,
    world: WorldModel,
    collision_geometry: Literal["line", "point"],
):
    """Create the appropriate IK solver based on the solver type."""
    if solver_type == "jax":
        from numeric_jax import IKNumericJAX

        return IKNumericJAX(
            model,
            world=world,
            collision_geometry=collision_geometry,
            max_iterations=200,
            lr=0.02,
            momentum=0.1,
        )
    elif solver_type == "torch":
        from numeric_torch import IKNumericTorch

        return IKNumericTorch(
            model,
            world=world,
            collision_geometry=collision_geometry,
            max_iterations=200,
            lr=0.06,
        )
    elif solver_type == "sympy":
        from numeric_sympy import IKNumericSympy

        return IKNumericSympy(
            model,
            world=world,
            collision_geometry=collision_geometry,
        )
    elif solver_type == "fabrik":
        from fabrik import IKFabrik

        return IKFabrik(
            model,
            world=world,
            collision_geometry=collision_geometry,
            max_iterations=100,
            tolerance=1e-6,
        )
    elif solver_type == "symbolic":
        from symbolic import IKSymbolic

        return IKSymbolic(model, world=world)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive IK solver demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--solver",
        "-s",
        choices=["jax", "torch", "sympy", "fabrik", "symbolic"],
        default="jax",
        help="Solver backend to use",
    )
    parser.add_argument(
        "--collision",
        "-c",
        choices=["line", "point"],
        default="line",
        help="Collision geometry type",
    )
    parser.add_argument(
        "--no-nogo",
        action="store_true",
        help="Disable nogo zones",
    )
    args = parser.parse_args()

    # Create a 3-link robot with joint limits
    model = RobotModel(
        link_lengths=(1.0, 0.8, 0.6),
        joint_limits=(
            (0.2 * math.pi, math.pi),
            (-math.pi, 0),
            (-math.pi / 2, math.pi / 2),
        ),
    )

    # Create a world with nogo zones (narrow corridor)
    if args.no_nogo:
        world = WorldModel(nogo=())
    else:
        nogo = (
            RegionHalfspace((0, -1), (0, -0.2)),
            RegionRectangle(0.5, 10.0, -10.0, 0.6),
            RegionRectangle(0.5, 10.0, 1.2, 5.0),
        )
        world = WorldModel(nogo=nogo)

    # Create the IK solver
    print(f"Creating {args.solver} solver with {args.collision} collision...")
    ik_solver = create_solver(args.solver, model, world, args.collision)

    # Initial position (within joint limits)
    initial_position = RobotPosition(joint_angles=(0.5 * math.pi, -math.pi / 4, 0.0))
    current_state = RobotState(model, current=initial_position, world=world)

    # Create visualizer
    viz = RobotVisualizer(current_state)

    # Click callback that updates the target and solves IK
    def on_click(x: float, y: float, btn: Literal["left", "right"]):
        nonlocal current_state
        print(f"\nClicked at: ({x:.2f}, {y:.2f}) {btn}")

        new_ee_angle: Optional[float] = (
            current_state.desired.ee_angle if current_state.desired else None
        )
        if btn == "right":
            new_ee_angle = 0.0 if new_ee_angle is None else None

        # Solve IK (with profiling if supported)
        try:
            # sympy solver doesn't support profiling
            result = ik_solver(
                current_state,
                DesiredPosition(ee_position=(x, y), ee_angle=new_ee_angle),
            )
            solution = result.state
            print(
                f"Solution: {tuple(f'{a:.3f}' for a in solution.current.joint_angles)}"
            )
            print(f"Solve time: {result.solve_time_ms:.2f}ms")
            print(f"Iterations: {result.iterations} (converged: {result.converged})")
            print(f"Loss: {result.initial_loss:.6f} -> {result.final_loss:.6f}")
            print(f"Position error: {result.position_error:.6f}")

            # Update the visualization with the new solution
            current_state = solution
            viz.update(current_state)

        except Exception as e:
            print(f"Error solving IK: {e}")
            import traceback

            traceback.print_exc()

    # Set the click callback
    viz.set_click_callback(on_click)

    print(f"\nInteractive IK Solver ({args.solver.upper()})")
    print("=" * 60)
    print(f"Solver: {args.solver}, Collision: {args.collision}")
    print("Left-click: Set target position")
    print("Right-click: Toggle angle constraint (locks to 0)")
    print("=" * 60)

    # Show the visualization
    viz.show()


if __name__ == "__main__":
    main()
