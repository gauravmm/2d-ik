# 2D Inverse Kinematics for 3D Printer Loading robot

This repository implements and validates a simple IK solver and motion planner for a pseudo-2D robot arm that exists to load and unload plates from a 3d printer.


## Technical architecture

We begin with these key assumptions:

1. Simple world geometry/exclusion zones.
2. No dynamic self-intersection (due to nested design on robot hardware)
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
    4. evaluate these for correctness and performance


## AI Use

All this is implemented with extensive use of Anthropic's Claude 4.5. A key purpose of this is to evaluate the use of AI on a simple (Undergrad-level) IK problem requring generalization from its training corpus.
