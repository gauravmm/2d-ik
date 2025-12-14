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
from dataclasses import dataclass
from typing import List, Tuple

from datamodel import RobotModel, RobotPosition, RobotState
from symbolic import IKSymbolic


@dataclass
class TimingResults:
    """Container for timing test results.

    Attributes:
        solver_init_time: Time to initialize IKSymbolic solver (seconds)
        warmup_times: Individual warmup solve times (seconds)
        solve_times: Individual solve times for grid points (seconds)
        grid_points: Target points tested
        robot_config: Link lengths of robot
    """
    solver_init_time: float
    warmup_times: List[float]
    solve_times: List[float]
    grid_points: List[Tuple[float, float]]
    robot_config: Tuple[float, ...]

    @property
    def mean_solve_time(self) -> float:
        """Mean solve time across all grid points."""
        return statistics.mean(self.solve_times) if self.solve_times else 0.0

    @property
    def median_solve_time(self) -> float:
        """Median solve time."""
        return statistics.median(self.solve_times) if self.solve_times else 0.0

    @property
    def std_dev(self) -> float:
        """Standard deviation of solve times."""
        return statistics.stdev(self.solve_times) if len(self.solve_times) > 1 else 0.0

    @property
    def min_time(self) -> float:
        """Minimum solve time."""
        return min(self.solve_times) if self.solve_times else 0.0

    @property
    def max_time(self) -> float:
        """Maximum solve time."""
        return max(self.solve_times) if self.solve_times else 0.0

    @property
    def p95(self) -> float:
        """95th percentile solve time."""
        if len(self.solve_times) < 20:
            return self.max_time
        sorted_times = sorted(self.solve_times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[idx]

    @property
    def p99(self) -> float:
        """99th percentile solve time."""
        if len(self.solve_times) < 100:
            return self.max_time
        sorted_times = sorted(self.solve_times)
        idx = int(len(sorted_times) * 0.99)
        return sorted_times[idx]

    @property
    def total_time(self) -> float:
        """Total time for all grid solves."""
        return sum(self.solve_times)


def calculate_workspace_bounds(
    model: RobotModel,
    coverage_ratio: float = 0.8
) -> Tuple[float, float, float, float]:
    """Calculate reasonable workspace bounds for a robot model.

    Args:
        model: The robot model
        coverage_ratio: Fraction of max reach to use (0.8 = 80%)

    Returns:
        (x_min, x_max, y_min, y_max) bounds
    """
    max_reach = sum(model.link_lengths)
    bound = max_reach * coverage_ratio
    return (-bound, bound, -bound, bound)


def generate_grid(
    n_points: int,
    bounds: Tuple[float, float, float, float],
) -> List[Tuple[float, float]]:
    """Generate a uniform grid of (x, y) points for testing.

    Args:
        n_points: Number of points per dimension (total = n_points^2)
        bounds: (x_min, x_max, y_min, y_max) workspace bounds

    Returns:
        List of (x, y) target positions
    """
    x_min, x_max, y_min, y_max = bounds

    # Generate linearly spaced points
    x_vals = [x_min + (x_max - x_min) * i / (n_points - 1) for i in range(n_points)]
    y_vals = [y_min + (y_max - y_min) * i / (n_points - 1) for i in range(n_points)]

    # Create grid
    grid_points = [(x, y) for x in x_vals for y in y_vals]

    return grid_points


def format_timing_summary(results: TimingResults) -> str:
    """Format timing results as a human-readable summary string.

    Args:
        results: TimingResults to format

    Returns:
        Formatted string with statistics
    """
    summary = []
    summary.append(f"\nTesting {len(results.robot_config)}-link robot {results.robot_config}")
    summary.append("=" * 80)
    summary.append(f"Solver Initialization: {results.solver_init_time:.3f}s")

    if results.warmup_times:
        warmup_mean = statistics.mean(results.warmup_times)
        summary.append(f"Warmup: {len(results.warmup_times)} iterations, mean: {warmup_mean:.4f}s")

    grid_size = int(math.sqrt(len(results.grid_points)))
    summary.append(f"Grid: {grid_size}x{grid_size} = {len(results.grid_points)} points")
    summary.append("")
    summary.append("Solve Time Statistics:")
    summary.append(f"  Mean:     {results.mean_solve_time:.4f}s")
    summary.append(f"  Median:   {results.median_solve_time:.4f}s")
    summary.append(f"  Std Dev:  {results.std_dev:.4f}s")
    summary.append(f"  Min:      {results.min_time:.4f}s")
    summary.append(f"  Max:      {results.max_time:.4f}s")
    summary.append(f"  P95:      {results.p95:.4f}s")
    summary.append(f"  P99:      {results.p99:.4f}s")
    summary.append("")
    summary.append(f"Total grid time: {results.total_time:.2f}s")
    summary.append(f"Amortized time per solve: {results.mean_solve_time:.4f}s")

    return "\n".join(summary)


def time_ik_grid_solve(
    link_lengths: Tuple[float, ...],
    grid_size: int = 15,
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
        TimingResults containing all timing data and statistics
    """
    # Step 1: Create robot model
    model = RobotModel(link_lengths=link_lengths)

    # Step 2: Time the solver initialization
    start_init = time.perf_counter()
    ik_solver = IKSymbolic(model)
    end_init = time.perf_counter()
    solver_init_time = end_init - start_init

    # Step 3: Calculate workspace bounds
    bounds = calculate_workspace_bounds(model, coverage_ratio)

    # Step 4: Generate grid of target points
    grid_points = generate_grid(grid_size, bounds)

    # Step 5: Perform warmup
    warmup_times = []
    if warmup_iterations > 0:
        # Use random subset of grid points for warmup
        warmup_points = random.sample(grid_points, min(warmup_iterations, len(grid_points)))

        for target in warmup_points:
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

    for target in grid_points:
        # Create zero initial position for each solve
        zero_position = RobotPosition(joint_angles=tuple(0.0 for _ in link_lengths))
        state = RobotState(model, zero_position, target)

        # Time the solve operation
        start = time.perf_counter()
        ik_solver(state)
        end = time.perf_counter()

        solve_times.append(end - start)

    # Step 7: Return results
    return TimingResults(
        solver_init_time=solver_init_time,
        warmup_times=warmup_times,
        solve_times=solve_times,
        grid_points=grid_points,
        robot_config=link_lengths,
    )


def test_three_link_timing():
    """Test timing for a three-link robot configuration (required test)."""
    results = time_ik_grid_solve(
        link_lengths=(1.0, 0.8, 0.6),
        grid_size=15,
        warmup_iterations=5,
    )

    print(format_timing_summary(results))

    # Assertions for reasonable performance
    assert results.mean_solve_time < 0.1, f"Mean solve time {results.mean_solve_time:.3f}s exceeds 100ms"
    assert results.solver_init_time < 10.0, f"Solver init time {results.solver_init_time:.3f}s exceeds 10s"
    assert len(results.solve_times) == 225, "Expected 225 grid points"


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


def test_solver_initialization_scaling():
    """Test how solver initialization time scales with robot complexity."""
    print("\n" + "=" * 80)
    print("Solver Initialization Scaling")
    print("=" * 80)

    configs = [
        (1.0, 1.0),
        (1.0, 0.8, 0.6),
        (1.0, 0.8, 0.6, 0.4),
        (1.0, 0.8, 0.6, 0.4, 0.3),
    ]

    print(f"\n{'Config':<20} {'Init Time (s)':<15}")
    print("-" * 35)

    for link_lengths in configs:
        model = RobotModel(link_lengths=link_lengths)

        start = time.perf_counter()
        IKSymbolic(model)
        end = time.perf_counter()

        init_time = end - start
        print(f"{len(link_lengths)}-link{'':<13} {init_time:>10.3f}s")


def test_warmup_effect():
    """Analyze the effect of warmup on solve times."""
    results = time_ik_grid_solve(
        link_lengths=(1.0, 0.8, 0.6),
        grid_size=10,
        warmup_iterations=5,
    )

    print("\n" + "=" * 80)
    print("Warmup Effect Analysis")
    print("=" * 80)

    if results.warmup_times:
        warmup_mean = statistics.mean(results.warmup_times)
        solve_mean = results.mean_solve_time

        print(f"\nWarmup mean:       {warmup_mean:.4f}s")
        print(f"Grid solve mean:   {solve_mean:.4f}s")
        print(f"Difference:        {(warmup_mean - solve_mean):.4f}s")

        if warmup_mean > solve_mean:
            improvement_pct = ((warmup_mean - solve_mean) / warmup_mean) * 100
            print(f"Improvement:       {improvement_pct:.1f}% faster after warmup")
        else:
            print("No significant warmup effect detected")


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
    unreachable_points = generate_grid(
        n_points=10,
        bounds=(-unreachable_bound, unreachable_bound, -unreachable_bound, unreachable_bound)
    )

    ik_solver = IKSymbolic(model)
    zero_position = RobotPosition(joint_angles=(0.0, 0.0, 0.0))

    unreachable_times = []
    for target in unreachable_points:
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
    reachable_mean = reachable_results.mean_solve_time

    print(f"\nReachable targets mean:     {reachable_mean:.4f}s")
    print(f"Unreachable targets mean:   {unreachable_mean:.4f}s")
    print(f"Ratio (unreachable/reach):  {unreachable_mean/reachable_mean:.2f}x")


def test_grid_density_comparison():
    """Compare performance with different grid densities."""
    print("\n" + "=" * 80)
    print("Grid Density Comparison")
    print("=" * 80)

    print(f"\n{'Grid Size':<12} {'Points':<8} {'Mean Time':<12} {'Total Time':<12}")
    print("-" * 44)

    for grid_size in [5, 10, 15, 20]:
        results = time_ik_grid_solve(
            link_lengths=(1.0, 0.8, 0.6),
            grid_size=grid_size,
            warmup_iterations=3,
        )

        print(f"{grid_size}x{grid_size:<8} {len(results.solve_times):<8} "
              f"{results.mean_solve_time:.4f}s{'':<3} {results.total_time:.2f}s")


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

        print(f"{len(link_lengths)}-link{'':<6} "
              f"{results.mean_solve_time:.4f}s{'':<3} "
              f"{results.median_solve_time:.4f}s{'':<5} "
              f"{results.p95:.4f}s")


if __name__ == "__main__":
    print("Running timing tests...")
    test_three_link_timing()
