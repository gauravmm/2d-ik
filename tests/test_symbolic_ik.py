#!/usr/bin/env python3
"""Simple test for symbolic IK solver."""

import math
from typing import List
from datamodel import DesiredPosition, RobotModel, RobotPosition, RobotState
from symbolic import IKSymbolic


def test_simple_two_link():
    """Test IK on a simple 2-link robot."""
    print("=" * 60)
    print("Test 1: Simple 2-link robot")
    print("=" * 60)

    # Create a 2-link robot with equal length links
    model = RobotModel(link_lengths=(1.0, 1.0))

    # Create the IK solver
    ik_solver = IKSymbolic(model)

    # Start from a known configuration
    initial_position = RobotPosition(joint_angles=(0.0, 0.0))

    # Target position: (1.5, 0.5) - should be reachable
    target = (1.5, 0.5)
    state = RobotState(model, initial_position)

    # Solve IK
    print(f"Target position: {target}")
    print(f"Initial joint angles: {initial_position.joint_angles}")

    solution_state = ik_solver(state, DesiredPosition(ee_position=target))
    solution = solution_state.current
    print(f"Solution joint angles: {solution.joint_angles}")

    # Verify the solution by computing forward kinematics
    end_effector_positions = model.forward_kinematics(solution)
    end_effector = end_effector_positions[-1]
    print(f"Achieved end effector position: {end_effector}")

    # Check error
    error = math.sqrt(
        (end_effector[0] - target[0]) ** 2 + (end_effector[1] - target[1]) ** 2
    )
    print(f"Position error: {error:.6f}")

    if error < 1e-3:
        print("✓ Test 1 passed!\n")
        return True
    else:
        print("✗ Test 1 failed - position error too large\n")
        return False


def test_three_link_robot():
    """Test IK on a 3-link robot."""
    print("=" * 60)
    print("Test 2: 3-link robot")
    print("=" * 60)

    # Create a 3-link robot
    model = RobotModel(link_lengths=(1.0, 0.8, 0.6))

    # Create the IK solver
    ik_solver = IKSymbolic(model)

    # Start from a configuration
    initial_position = RobotPosition(joint_angles=(0.0, math.pi / 4, -math.pi / 4))

    # Target position
    target = (1.8, 0.8)
    state = RobotState(model, initial_position)

    # Solve IK
    print(f"Target position: {target}")
    print(f"Initial joint angles: {initial_position.joint_angles}")

    solution_state = ik_solver(state, DesiredPosition(ee_position=target))
    solution = solution_state.current
    print(f"Solution joint angles: {solution.joint_angles}")

    # Verify the solution
    end_effector_positions = model.forward_kinematics(solution)
    end_effector = end_effector_positions[-1]
    print(f"Achieved end effector position: {end_effector}")

    # Check error
    error = math.sqrt(
        (end_effector[0] - target[0]) ** 2 + (end_effector[1] - target[1]) ** 2
    )
    print(f"Position error: {error:.6f}")

    if error < 1e-3:
        print("✓ Test 2 passed!\n")
        return True
    else:
        print("✗ Test 2 failed - position error too large\n")
        return False


def test_unreachable_target():
    """Test IK with a target outside robot's reach."""
    print("=" * 60)
    print("Test 3: Unreachable target (best effort)")
    print("=" * 60)

    # Create a 2-link robot
    model = RobotModel(link_lengths=(1.0, 1.0))

    # Create the IK solver
    ik_solver = IKSymbolic(model)

    # Start from a configuration
    initial_position = RobotPosition(joint_angles=(0.0, 0.0))

    # Target position: (3.0, 0.0) - outside max reach of 2.0
    target = (3.0, 0.0)
    state = RobotState(model, initial_position)

    # Solve IK
    print(f"Target position: {target}")
    print(f"Robot max reach: {sum(model.link_lengths)}")
    print(f"Initial joint angles: {initial_position.joint_angles}")

    solution_state = ik_solver(state, DesiredPosition(ee_position=target))
    solution = solution_state.current
    print(f"Solution joint angles: {solution.joint_angles}")

    # Verify the solution
    end_effector_positions = model.forward_kinematics(solution)
    end_effector = end_effector_positions[-1]
    print(f"Achieved end effector position: {end_effector}")

    # Check that we got as close as possible
    distance_to_target = math.sqrt(
        (end_effector[0] - target[0]) ** 2 + (end_effector[1] - target[1]) ** 2
    )
    distance_from_origin = math.sqrt(end_effector[0] ** 2 + end_effector[1] ** 2)
    max_reach = sum(model.link_lengths)

    print(f"Distance to target: {distance_to_target:.6f}")
    print(f"Distance from origin: {distance_from_origin:.6f}")

    # Should be at max reach (arms fully extended)
    if abs(distance_from_origin - max_reach) < 1e-2:
        print("✓ Test 3 passed! (Robot extended to max reach)\n")
        return True
    else:
        print("✗ Test 3 failed - robot not at max reach\n")
        return False


if __name__ == "__main__":
    test_results : List[bool] = []
    test_results.append(test_simple_two_link())
    test_results.append(test_three_link_robot())
    test_results.append(test_unreachable_target())

    print("=" * 60)
    print(f"Results: {sum(test_results)}/{len(test_results)} tests passed")
    print("=" * 60)

    exit(0 if all(test_results) else 1)
