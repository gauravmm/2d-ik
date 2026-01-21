#!/usr/bin/env python3
"""Comprehensive timing tests for 2D IK solver.

This module provides performance testing for the symbolic IK solver, measuring:
- Solver initialization time
- Single solve performance
- Grid-based batch solving with statistical analysis
- Performance scaling with robot complexity

The main helper function time_ik_grid_solve() constructs a robot model,
solves IK for a dense grid of points, and measures timing with warmup.
"""

import time
import statistics
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal

import matplotlib.figure
import numpy as np
import matplotlib.pyplot as plt

from datamodel import DesiredPosition, RobotModel, RobotPosition, RobotState
from symbolic import IKSymbolic


@dataclass(frozen=True)
class TimingResults:
    """Results from timing IK solver performance on a grid of points.

    Attributes:
        solver_init_time: Time to initialize IKSymbolic solver (seconds)
        warmup_times: Individual warmup solve times (seconds)
        solve_times: Individual solve times for grid points (seconds)
        grid_size: Number of points per dimension
        robot_config: Link lengths of robot
        grid_points: Tuple of (x_coords, y_coords) arrays for the grid
        solve_errors: Array of position errors for each grid point (distance from target)
    """

    solver_init_time: float
    warmup_times: List[float]
    solve_times: List[float]
    grid_size: int
    robot_config: Tuple[float, ...]
    grid_points: Tuple[np.ndarray, np.ndarray]
    solve_errors: List[float]


def plot_solve_heatmap(
    results: TimingResults, kind: Literal["error", "time"]
) -> matplotlib.figure.Figure:
    """Render a heatmap showing either the error or the time taken to solve for each grid position.

    Args:
        results: TimingResults dataclass containing timing and error data
        kind: Type of heatmap to generate - "error" for position errors, "time" for solve times

    Returns:
        matplotlib Figure object containing the heatmap
    """
    # Extract grid coordinates and reshape data to grid form
    x_coords, y_coords = results.grid_points
    grid_size = results.grid_size

    # Select data to plot based on kind
    if kind == "error":
        data = np.array(results.solve_errors).reshape(grid_size, grid_size)
        title = f"IK Solver Position Error - {len(results.robot_config)}-link robot"
        cbar_label = "Position Error (units)"
        cmap = "viridis"
    elif kind == "time":
        data = np.log(np.array(results.solve_times).reshape(grid_size, grid_size))
        title = f"IK Solver Time - {len(results.robot_config)}-link robot"
        cbar_label = "Solve Time (log s)"
        cmap = "plasma"
    else:
        raise ValueError(f"Invalid kind: {kind}. Must be 'error' or 'time'")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create heatmap using pcolormesh for better performance
    im = ax.imshow(
        data,
        cmap=cmap,
        aspect="equal",
        extent=(np.min(x_coords), np.max(x_coords), np.min(y_coords), np.max(y_coords)),
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, rotation=270, labelpad=20)

    # Set labels and title
    ax.set_xlabel("Target X Position")
    ax.set_ylabel("Target Y Position")
    ax.set_title(title)
    ax.set_aspect("equal")

    # Add grid
    ax.grid(True, alpha=0.3)

    return fig


def format_timing_summary(results: TimingResults) -> str:
    """Format timing results as a human-readable summary string.

    Args:
        results: TimingResults dataclass instance

    Returns:
        Formatted string with statistics
    """
    solver_init_time = results.solver_init_time
    warmup_times = results.warmup_times
    solve_times = results.solve_times
    grid_size = results.grid_size
    robot_config = results.robot_config

    # Calculate percentiles
    assert len(solve_times) >= 2

    sorted_times = sorted(solve_times)
    p95 = p99 = max_time = sorted_times[-1]
    if len(solve_times) < 20:
        p95 = max_time
    else:
        sorted_times = sorted(solve_times)
        p95 = sorted_times[int(len(sorted_times) * 0.95)]

    if len(solve_times) < 100:
        p99 = max_time
    else:
        sorted_times = sorted(solve_times)
        p99 = sorted_times[int(len(sorted_times) * 0.99)]

    total_time = sum(solve_times)

    # Format output (convert to milliseconds)
    summary = f"""
Testing {len(robot_config)}-link robot {robot_config}
Solver Initialization: {solver_init_time * 1000:.2f}ms"""

    if warmup_times:
        summary += f"\nWarmup: {len(warmup_times)} iterations, max: {max(warmup_times) * 1000:.2f}ms"

    summary += f"""
Grid: {grid_size}x{grid_size} = {grid_size**2} points
Solve Time Statistics:
    Mean:     {statistics.mean(solve_times) * 1000:.2f}ms
    Median:   {statistics.median(solve_times) * 1000:.2f}ms
    Std Dev:  {statistics.stdev(solve_times) * 1000:.2f}ms
    Min:      {sorted_times[0] * 1000:.2f}ms
    Max:      {sorted_times[-1] * 1000:.2f}ms
    P95:      {p95 * 1000:.2f}ms
    P99:      {p99 * 1000:.2f}ms

Total grid time: {total_time:.2f}s
"""
    return summary


def time_ik_grid_solve(
    link_lengths: Tuple[float, ...],
    grid_size: int = 15,
    locked_ee_angle: Optional[float] = None,
    warmup_iterations: int = 5,
    coverage_ratio: float = 0.8,
) -> TimingResults:
    """Construct a robot model, solve IK for a dense grid of points, and measure timing.

    This is the core helper function that:
    1. Creates a RobotModel with specified link lengths
    2. Initializes IKSymbolic solver (timed)
    3. Generates dense grid of target points within workspace
    4. Performs warmup solves
    5. Measures time for each solve in the grid

    Each solve starts from zero position as required.

    Args:
        link_lengths: Tuple of link lengths for the robot
        grid_size: Number of points per dimension (total = grid_size^2)
        warmup_iterations: Number of warmup solves before measurement
        coverage_ratio: Fraction of max reach to test (0.8 = 80%)

    Returns:
        TimingResults dataclass containing timing data
    """
    # Step 1: Create robot model
    model = RobotModel(link_lengths=link_lengths)

    # Step 2: Time the solver initialization
    start_init = time.perf_counter()
    ik_solver = IKSymbolic(model)
    end_init = time.perf_counter()
    solver_init_time = end_init - start_init

    # Step 3: Calculate workspace bounds
    bounds = sum(model.link_lengths) * coverage_ratio

    # Step 4: Generate grid of target points
    x_coords, y_coords = np.meshgrid(
        np.linspace(-bounds, bounds, grid_size),
        np.linspace(-bounds, bounds, grid_size),
    )
    x_coords = x_coords.ravel()
    y_coords = y_coords.ravel()
    n_points = len(x_coords)

    # Step 5: Perform warmup
    warmup_times = []
    if warmup_iterations > 0:
        # Use random subset of grid points for warmup
        warmup_indices = random.sample(
            range(n_points), min(warmup_iterations, n_points)
        )

        for idx in warmup_indices:
            target = (float(x_coords[idx]), float(y_coords[idx]))
            # Create zero initial position
            zero_position = RobotPosition(joint_angles=tuple(0.0 for _ in link_lengths))
            state = RobotState(model, zero_position)

            # Time the solve
            start = time.perf_counter()
            ik_solver(
                state, DesiredPosition(ee_position=target, ee_angle=locked_ee_angle)
            )
            end = time.perf_counter()

            warmup_times.append(end - start)

    # Step 6: Measure grid solves
    solve_times = []
    solve_errors = []
    zero_position = RobotPosition(joint_angles=tuple(0.0 for _ in link_lengths))

    for target in zip(x_coords, y_coords):
        state = RobotState(model, zero_position)

        # Time the solve operation
        start = time.perf_counter()
        solution_state = ik_solver(
            state, DesiredPosition(ee_position=target, ee_angle=locked_ee_angle)
        )
        end = time.perf_counter()

        solve_times.append(end - start)

        # Calculate position error
        solution = solution_state.current
        end_effector_positions = model.forward_kinematics(solution)
        end_effector = end_effector_positions[-1]
        error = math.sqrt(
            (end_effector[0] - target[0]) ** 2 + (end_effector[1] - target[1]) ** 2
        )
        solve_errors.append(error)

    # Step 7: Return results
    return TimingResults(
        solver_init_time=solver_init_time,
        warmup_times=warmup_times,
        solve_times=solve_times,
        grid_size=grid_size,
        robot_config=link_lengths,
        grid_points=(x_coords, y_coords),
        solve_errors=solve_errors,
    )


def test_three_link_timing(
    link_lengths=(1.0, 0.8, 0.6), grid_size=15, locked_ee_angle: Optional[float] = None
):
    """Test timing for a three-link robot configuration (required test)."""
    results = time_ik_grid_solve(
        link_lengths=link_lengths,
        grid_size=grid_size,
        locked_ee_angle=locked_ee_angle,
        warmup_iterations=5,
    )

    print(format_timing_summary(results))

    # Assertions for reasonable performance
    mean_solve = statistics.mean(results.solve_times)
    assert mean_solve < 0.01, f"Mean solve time {mean_solve:.3f}s exceeds 10ms"
    assert results.solver_init_time < 10.0, (
        f"Solver init time {results.solver_init_time:.3f}s exceeds 10s"
    )
    assert len(results.solve_times) == grid_size**2, "Expected 225 grid points"

    return results


def evaluate_robot_complexity_scaling():
    """Measure how solve time scales with robot complexity."""
    print("\n" + "=" * 80)
    print("Robot Complexity Scaling")
    print("=" * 80)

    configs = [tuple([1.0] * k) for k in range(2, 8)]

    def print_cols(*parts: str, width: int = 12):
        print("".join(p.rjust(width) for p in parts))

    print_cols("Config", "Warmup", "Mean", "Median", "P95")
    print("-" * 60)

    for link_lengths in configs:
        results = time_ik_grid_solve(
            link_lengths=link_lengths,
            grid_size=10,
            warmup_iterations=3,
        )

        solve_times = results.solve_times
        sorted_times = sorted(solve_times)
        p95 = (
            sorted_times[int(len(sorted_times) * 0.95)]
            if len(sorted_times) >= 20
            else sorted_times[-1]
        )

        print_cols(
            f"{len(link_lengths)}-link",
            f"{max(results.warmup_times) * 1000:.2f}ms",
            f"{statistics.mean(solve_times) * 1000:.2f}ms",
            f"{statistics.median(solve_times) * 1000:.2f}ms",
            f"{p95 * 1000:.2f}ms",
        )


if __name__ == "__main__":
    print("Running timing tests...")
    # evaluate_robot_complexity_scaling()
    results = test_three_link_timing(grid_size=50, locked_ee_angle=0.0)

    fig = plot_solve_heatmap(results, "error")
    fig.show()
    plt.show()
