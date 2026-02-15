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


def _animation_checkpoints(model):
    checkpoints = [(2.5, 1), (2.5, 0.7), (0, 0.7), (0, 0.2), (0, 2), (0, 1)]
    return [
        DesiredPosition(ee_position=(x, y), ee_angle=ee_angle)
        for x, y in checkpoints
        for ee_angle in (0.0, None)
    ]


def run_animation(args):
    """Run the animation mode, following checkpoints with smooth position interpolation."""
    model = RobotModel(
        link_lengths=(1.0, 0.8, 0.6),
        joint_limits=(
            (0.2 * math.pi, math.pi),
            (-math.pi, 0),
            (-math.pi * 0.9, math.pi * 0.9),
        ),
    )

    if args.no_nogo:
        world = WorldModel(nogo=())
    else:
        nogo = (
            RegionHalfspace((0, -1), (0, -0.2)),
            RegionRectangle(0.5, 10.0, -10.0, 0.6),
            RegionRectangle(0.5, 10.0, 1.2, 5.0),
        )
        world = WorldModel(nogo=nogo)

    print(f"Creating {args.solver} solver with {args.collision} collision...")
    ik_solver = create_solver(args.solver, model, world, args.collision)

    initial_position = RobotPosition(joint_angles=(0.5 * math.pi, -math.pi / 4, 0.0))
    current_state = RobotState(model, current=initial_position, world=world)

    checkpoints = _animation_checkpoints(model)
    max_step = 0.05  # Maximum distance between intermediate targets

    # Build trajectory: subdivide each segment and solve IK at every step
    print("Building trajectory...")
    solved_frames: list[RobotState] = [current_state]
    state = current_state

    for i, cp in enumerate(checkpoints):
        # Get start position from current state
        if state.desired and state.desired.ee_position:
            start_pos = state.desired.ee_position
        else:
            # Use forward kinematics end effector position
            positions = state.get_joint_positions()
            start_pos = positions[-1]

        end_pos = cp.ee_position
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        dist = math.hypot(dx, dy)
        num_steps = max(1, math.ceil(dist / max_step))

        # Angle changes suddenly at the start of each segment
        ee_angle = cp.ee_angle

        converged = True
        for step in range(1, num_steps + 1):
            t = step / num_steps
            interp_pos = (
                start_pos[0] + dx * t,
                start_pos[1] + dy * t,
            )
            desired = DesiredPosition(ee_position=interp_pos, ee_angle=ee_angle)
            result = ik_solver(state, desired)
            state = result.state
            converged = converged and result.converged
            solved_frames.append(state)

        print(
            f"  Checkpoint {i}: pos=({cp.ee_position[0]:.1f}, {cp.ee_position[1]:.1f}), "
            f"angle={'None' if cp.ee_angle is None else f'{cp.ee_angle:.1f}'}, "
            f"steps={num_steps}, converged={converged}"
        )

    print(f"Total frames: {len(solved_frames)}")

    def update_func(frame: int) -> RobotState:
        return solved_frames[min(frame, len(solved_frames) - 1)]

    viz = RobotVisualizer(current_state)

    print(f"\nAnimation Mode ({args.solver.upper()})")
    print("=" * 60)
    print(f"Checkpoints: {len(checkpoints)}, Total frames: {len(solved_frames)}")
    print("=" * 60)

    viz.animate(update_func, interval=33, frames=len(solved_frames))


def main(args):
    # Create a 3-link robot with joint limits
    model = RobotModel(
        link_lengths=(1.0, 0.8, 0.6),
        joint_limits=(
            (0.2 * math.pi, math.pi),
            (-math.pi, 0),
            (-math.pi * 0.9, math.pi * 0.9),
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
            if result.initial_loss < 0:
                print(f"Loss: {result.final_loss:.6f}")
            else:
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
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Run animation following pre-defined checkpoints",
    )
    args = parser.parse_args()

    if args.animate:
        run_animation(args)
    else:
        main(args)
