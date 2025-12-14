# 2D Inverse Kinematics for 3D Printer Loading robot

This repository implements and validates a simple IK solver and motion planner for a pseudo-2D robot arm that exists to load and unload plates from a 3d printer.

## Technical architecture

We begin with these key assumptions:

1. Very simple robot design
    a. No dynamic self-intersection (nested robot design)
    b. all rigid links are collinear (can be represented as a line)
2. Simple world geometry/exclusion zones.
3. Quasi-static motion (i.e. motion can be paused at any time.)
4. End effector must be controllable in both position and orientation.

To achieve this, we write:

1. a super-simple data model
2. simulator, including intersection checking,
3. three solvers:
    a `sympy`-based,
    b. jacobian-based, and
    c. a FABRIK-based IK solver
4. a simple trajectory interpolator that "chops up" the overall motion into segments:
    a. subject to maximum constraints
    b. avoiding collisions
5. evaluate these for correctness and performance

## Usage

### Interactive IK Solver Demo

Run the interactive solver to visualize and test the IK system:

```bash
source ~/.pyenv/versions/ik2d/bin/activate
python symbolic.py
```

This will open a matplotlib window with a 3-link robot arm. Click anywhere in the window to set a target position, and the robot will solve inverse kinematics to reach that point. The green circle shows the target position, and the console displays the solution joint angles and position error.

### Running Tests

Test the forward kinematics implementation:

```bash
source ~/.pyenv/versions/ik2d/bin/activate
pytest test_datamodel.py
```

Test the symbolic IK solver:

```bash
source ~/.pyenv/versions/ik2d/bin/activate
python test_symbolic_ik.py
```

## AI Use

All this is implemented with extensive use of Anthropic's Claude 4.5. A key purpose of this is to evaluate the use of AI on a simple (Undergrad-level) IK problem requring generalization from its training corpus.

I wrote this with Claude Sonnet 4.5, which is incredible! It required a little strategic prompting, but generally acted like a mid-level developer with a caffeine addiction. It came up with the data model and forward kinematics easily, and was able to debug issues with the visualization code much quicker than me. (The issue it both caused and debugged was that matplotlib's animation functions require the artists to be drawn on first initialization.)

## Instructions for AI Agent

Activate the pyenv environment `ik2d` before running any code.
