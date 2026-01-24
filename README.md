# 2D Inverse Kinematics for 3D Printer Loading robot

This repository implements and validates a simple IK solver and motion planner for a pseudo-2D robot arm that exists to load and unload plates from a 3d printer.

## Technical architecture

We begin with these key assumptions:

1. Very simple robot design
    a. No dynamic self-intersection (nested robot design)
    b. all rigid links can be represented as a line.
2. Simple world geometry/exclusion zones.
3. Quasi-static motion (i.e. motion can be paused at any time.)
4. End effector must be controllable in both position and orientation.

To achieve this, we write:

1. a super-simple data model
2. simulator, including intersection checking,
3. five solvers:
    a. `jax` jacobian-based with precompilation,
    b. `torch` jacobian-based with dynamic computation graph,
    c `sympy`-based numerical optimizer,
    d `sympy`-based symbolic solver (doesn't work), and
    e. a FABRIK-based IK solver

## Usage

### Interactive Demo

Run the interactive solver to visualize and test the IK system:

```bash
uv run python demo.py --solver {jax,torch,sympy,fabrik,symbolic}
```

This will open a matplotlib window with a 3-link robot arm. Click anywhere in the window to set a target position, and the robot will move towards that point avoiding the red no-go zones. The green circle shows the target position, and the console displays the solution joint angles and position error.

The best solver is `jax`, which optimizes an objective using gradient descent to find joint angles subject to constraints and arm geometry. (This is similar to the "jacobian" method in IK literature.) `torch` does the same with PyTorch, but substantially slower. `fabrik` uses the relatively new heuristic algorithm with a new projection idea to navigate no-go zones.

The `symbolic` and `sympy` solvers use the [SymPy](https://www.sympy.org/en/index.html) symbolic mathematics library to build the equations for the robot. `symbolic` solves these analytically (or fails to), and `sympy` runs a simple numerical optimizer to solve it.

### Running Tests

For proper directory import run this:

```bash
uv run python -m pytest
```

## AI Use

All this is implemented with extensive use of Anthropic's Claude 4.5. A key purpose of this is to evaluate the use of AI on a simple (Undergrad-level) IK problem requring generalization from its training corpus.

I wrote this with Claude Sonnet 4.5, which is incredible! It required a little strategic prompting, but generally acted like a mid-level developer with a caffeine addiction. It came up with the data model and forward kinematics easily, and was able to debug issues relatively reliably. It occasionally stumbled (matplotlib animations without rendering all artists in the initialization phase, creating unnecessary intermediate data types for `jax`, etc.) but was easily prompted into correcting its mistakes.
