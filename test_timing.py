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

try:
    import pytest
except ImportError:
    pytest = None

import time
import statistics
import math
import random
from typing import Dict, List, Tuple, Any

import numpy as np

from datamodel import RobotModel, RobotPosition, RobotState
from symbolic import IKSymbolic


def format_timing_summary(
    solver_init_time: float,
    warmup_times: List[float],
    solve_times: List[float],
    grid_size: int,
    robot_config: Tuple[float, ...],
) -> str:
    """Format timing results as a human-readable summary string.

    Args:
        solver_init_time: Time to initialize IKSymbolic solver (seconds)
        warmup_times: Individual warmup solve times (seconds)
        solve_times: Individual solve times for grid points (seconds)
        grid_size: Number of points per dimension
        robot_config: Link lengths of robot

    Returns:
        Formatted string with statistics
    """
    # Calculate statistics
    mean_solve = statistics.mean(solve_times) if solve_times else 0.0
    median_solve = statistics.median(solve_times) if solve_times else 0.0
    std_dev = statistics.stdev(solve_times) if len(solve_times) > 1 else 0.0
    min_time = min(solve_times) if solve_times else 0.0
    max_time = max(solve_times) if solve_times else 0.0

    # Calculate percentiles
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

    # Format output
    summary = []
    summary.append(f"\nTesting {len(robot_config)}-link robot {robot_config}")
    summary.append("=" * 80)
    summary.append(f"Solver Initialization: {solver_init_time:.3f}s")

    if warmup_times:
        warmup_mean = statistics.mean(warmup_times)
        summary.append(
            f"Warmup: {len(warmup_times)} iterations, mean: {warmup_mean:.4f}s"
        )

    summary.append(f"Grid: {grid_size}x{grid_size} = {grid_size**2} points")
    summary.append("")
    summary.append("Solve Time Statistics:")
    summary.append(f"  Mean:     {mean_solve:.4f}s")
    summary.append(f"  Median:   {median_solve:.4f}s")
    summary.append(f"  Std Dev:  {std_dev:.4f}s")
    summary.append(f"  Min:      {min_time:.4f}s")
    summary.append(f"  Max:      {max_time:.4f}s")
    summary.append(f"  P95:      {p95:.4f}s")
    summary.append(f"  P99:      {p99:.4f}s")
    summary.append("")
    summary.append(f"Total grid time: {total_time:.2f}s")
    summary.append(f"Amortized time per solve: {mean_solve:.4f}s")

    return "\n".join(summary)


def time_ik_grid_solve(
    link_lengths: Tuple[float, ...],
    grid_size: int = 15,
    warmup_iterations: int = 5,
    coverage_ratio: float = 0.8,
) -> Dict[str, Any]:
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
        Dictionary containing timing data:
        - solver_init_time: Time to initialize solver (seconds)
        - warmup_times: List of warmup solve times (seconds)
        - solve_times: List of grid solve times (seconds)
        - grid_size: Grid size (n_points per dimension)
        - robot_config: Tuple of link lengths
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
        np.linspace(-bounds, -bounds, n_points), np.linspace(-bounds, -bounds, n_points)
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
            state = RobotState(model, zero_position, target)

            # Time the solve
            start = time.perf_counter()
            ik_solver(state)
            end = time.perf_counter()

            warmup_times.append(end - start)

    # Step 6: Measure grid solves
    solve_times = []
    zero_position = RobotPosition(joint_angles=tuple(0.0 for _ in link_lengths))

    for idx in range(n_points):
        target = (float(x_coords[idx]), float(y_coords[idx]))
        state = RobotState(model, zero_position, target)

        # Time the solve operation
        start = time.perf_counter()
        ik_solver(state)
        end = time.perf_counter()

        solve_times.append(end - start)

    # Step 7: Return results
    return {
        "solver_init_time": solver_init_time,
        "warmup_times": warmup_times,
        "solve_times": solve_times,
        "grid_size": grid_size,
        "robot_config": link_lengths,
    }


def test_three_link_timing():
    """Test timing for a three-link robot configuration (required test)."""
    results = time_ik_grid_solve(
        link_lengths=(1.0, 0.8, 0.6),
        grid_size=15,
        warmup_iterations=5,
    )

    print(format_timing_summary(**results))

    # Assertions for reasonable performance
    mean_solve = statistics.mean(results["solve_times"])
    assert mean_solve < 0.1, f"Mean solve time {mean_solve:.3f}s exceeds 100ms"
    assert (
        results["solver_init_time"] < 10.0
    ), f"Solver init time {results['solver_init_time']:.3f}s exceeds 10s"
    assert len(results["solve_times"]) == 15**2, "Expected 225 grid points"


def test_single_solve_timing():
    """Measure timing for a single IK solve with different target scenarios."""
    print("\n" + "=" * 80)
    print("Single Solve Timing Comparison")
    print("=" * 80)

    # Create a 3-link robot
    model = RobotModel(link_lengths=(1.0, 0.8, 0.6))
    ik_solver = IKSymbolic(model)
    zero_position = RobotPosition(joint_angles=(0.0, 0.0, 0.0))

    # Test scenarios
    scenarios = [
        ("Center (easy)", (1.0, 0.0)),
        ("Near max reach", (2.2, 0.5)),
        ("Reachable target", (1.5, 1.0)),
    ]

    print(f"\n{'Scenario':<20} {'Time (ms)':<12}")
    print("-" * 32)

    for name, target in scenarios:
        state = RobotState(model, zero_position, target)

        start = time.perf_counter()
        ik_solver(state)
        end = time.perf_counter()

        solve_time_ms = (end - start) * 1000
        print(f"{name:<20} {solve_time_ms:>8.2f}ms")


def test_unreachable_targets_timing():
    """Measure timing for unreachable targets (worst-case)."""
    print("\n" + "=" * 80)
    print("Unreachable Targets Timing (Worst-case)")
    print("=" * 80)

    # Create a 3-link robot
    model = RobotModel(link_lengths=(1.0, 0.8, 0.6))
    max_reach = sum(model.link_lengths)

    # Generate points outside max reach (1.2x max reach)
    unreachable_bound = max_reach * 1.2
    x_coords, y_coords = generate_grid(
        n_points=10,
        bounds=(
            -unreachable_bound,
            unreachable_bound,
            -unreachable_bound,
            unreachable_bound,
        ),
    )

    ik_solver = IKSymbolic(model)
    zero_position = RobotPosition(joint_angles=(0.0, 0.0, 0.0))

    unreachable_times = []
    for idx in range(len(x_coords)):
        target = (float(x_coords[idx]), float(y_coords[idx]))
        state = RobotState(model, zero_position, target)

        start = time.perf_counter()
        ik_solver(state)
        end = time.perf_counter()

        unreachable_times.append(end - start)

    # Also test reachable targets for comparison
    reachable_results = time_ik_grid_solve(
        link_lengths=(1.0, 0.8, 0.6),
        grid_size=10,
        warmup_iterations=0,
        coverage_ratio=0.8,
    )

    unreachable_mean = statistics.mean(unreachable_times)
    reachable_mean = statistics.mean(reachable_results["solve_times"])

    print(f"\nReachable targets mean:     {reachable_mean:.4f}s")
    print(f"Unreachable targets mean:   {unreachable_mean:.4f}s")
    print(f"Ratio (unreachable/reach):  {unreachable_mean/reachable_mean:.2f}x")


def test_robot_complexity_scaling():
    """Measure how solve time scales with robot complexity."""
    print("\n" + "=" * 80)
    print("Robot Complexity Scaling")
    print("=" * 80)

    configs = [
        (1.0, 1.0),
        (1.0, 0.8, 0.6),
        (1.0, 0.8, 0.6, 0.4),
    ]

    print(f"\n{'Config':<12} {'Mean Time':<12} {'Median Time':<14} {'P95':<10}")
    print("-" * 48)

    for link_lengths in configs:
        results = time_ik_grid_solve(
            link_lengths=link_lengths,
            grid_size=10,
            warmup_iterations=3,
        )

        solve_times = results["solve_times"]
        mean_solve = statistics.mean(solve_times)
        median_solve = statistics.median(solve_times)
        sorted_times = sorted(solve_times)
        p95 = (
            sorted_times[int(len(sorted_times) * 0.95)]
            if len(sorted_times) >= 20
            else max(solve_times)
        )

        print(
            f"{len(link_lengths)}-link{'':<6} "
            f"{mean_solve:.4f}s{'':<3} "
            f"{median_solve:.4f}s{'':<5} "
            f"{p95:.4f}s"
        )


if __name__ == "__main__":
    print("Running timing tests...")
    test_three_link_timing()
