# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

2D Inverse Kinematics solver for a pseudo-2D robot arm designed for loading/unloading plates from a 3D printer. The project implements and validates IK solvers with motion planning capabilities.

## Development Environment

This project uses uv. Invoke code uring

```bash
uv run python ...
```

If you change more than a few lines of code, commit all changes you make with a short message.

### Running Tests

If you are running all tests, you should request the user to run them on your behalf.

## Architecture

### Core Data Model (datamodel.py)

Three immutable dataclasses define the robot:

- **RobotModel**: Defines robot geometry with `link_lengths` (tuple of floats) and optional `joint_origins` (initial angle offsets for each joint)
- **RobotPosition**: Contains `joint_angles` (tuple of floats in radians, relative to joint origins)
- **RobotState**: Combines RobotModel, current RobotPosition, and optional desired end effector position/angle

**Forward kinematics** is implemented in `RobotModel.forward_kinematics()`, which takes a RobotPosition and returns a list of (x, y) tuples for each joint position (starting with base at origin).

### IK Solver (symbolic.py)

**IKSymbolic** class implements inverse kinematics using symbolic mathematics:

1. Uses SymPy to build symbolic forward kinematics equations
2. Derives analytical gradient of combined objective function (position error + optional weighted angle error)
3. Compiles symbolic expressions to fast numerical functions via `sp.lambdify`
4. Optimizes using scipy's BFGS with analytical gradient

**Angle constraints**: When `RobotState.desired_end_effector_angle` is set, the solver constrains the end effector orientation using a weighted angle error term (weight = 1.0e3). When None, only position is optimized.

Key implementation detail: The angle error uses `atan2(sin(diff), cos(diff))` to ensure smooth wrapping to [-π, π].

### Visualization (visualization.py)

**RobotVisualizer** class provides interactive matplotlib visualization:

- Displays robot arm with links (blue lines) and joints (red circles)
- Green circle shows desired end effector position
- Supports click callbacks for interactive demos
- Auto-scales view based on robot reach
- Can animate robot motion via `animate()` method

The visualizer handles both left-click (set target position) and right-click (toggle angle constraint) events in the interactive demo.

## Project Assumptions

1. Simple robot design: no dynamic self-intersection, all rigid links collinear
2. Simple world geometry/exclusion zones
3. Quasi-static motion (can pause at any time)
4. End effector controllable in both position and orientation

## Recent Features

- Angle locking: End effector can be constrained to specific orientations
- Visualization of angle constraints in interactive demo
- Right-click interaction to toggle angle constraints
