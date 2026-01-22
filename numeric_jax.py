#!python3
"""JAX-based numerical IK solver with JIT-compiled core optimization loop."""

import time
from dataclasses import dataclass
from functools import partial
from typing import Literal, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax

from datamodel import (
    DesiredPosition,
    Region,
    RegionBall,
    RegionHalfspace,
    RegionRectangle,
    RobotModel,
    RobotPosition,
    RobotState,
    WorldModel,
)


# Type alias for JAX arrays
Array = jax.Array


class RobotParams(NamedTuple):
    """Static robot parameters for JIT compilation."""

    link_lengths: Array
    joint_origins: Array
    joint_min: Array  # Joint minimum limits (use -inf for unconstrained)
    joint_max: Array  # Joint maximum limits (use inf for unconstrained)


class NogoRegionParams(NamedTuple):
    """Parameters for nogo regions, stored as arrays for JIT compatibility."""

    # Halfspace regions: (n_halfspaces, 4) - [normal_x, normal_y, anchor_x, anchor_y]
    halfspaces: Array
    # Ball regions: (n_balls, 3) - [center_x, center_y, radius]
    balls: Array
    # Rectangle regions: (n_rects, 4) - [left, right, bottom, top]
    rectangles: Array


class SolverState(NamedTuple):
    """State for the optimization loop."""

    thetas: Array
    velocity: Array  # For momentum-based optimization
    iteration: int
    prev_loss: float
    converged: bool


class SolverResult(NamedTuple):
    """Result from the JIT-compiled solver."""

    thetas: Array
    iterations: int
    converged: bool
    final_loss: float


def _forward_kinematics(
    thetas: Array, link_lengths: Array, joint_origins: Array
) -> Tuple[Array, Array, Array, Array, Array]:
    """Compute forward kinematics using JAX.

    Returns:
        Tuple of (ee_x, ee_y, ee_angle, joint_xs, joint_ys)
        where joint_xs/joint_ys are arrays of all joint positions including origin.
    """
    n_joints = link_lengths.shape[0]

    def scan_fn(carry, inputs):
        x, y, cumulative_angle = carry
        link_length, joint_origin, theta = inputs

        cumulative_angle = cumulative_angle + theta + joint_origin
        x = x + link_length * jnp.cos(cumulative_angle)
        y = y + link_length * jnp.sin(cumulative_angle)

        return (x, y, cumulative_angle), (x, y)

    init_carry = (jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0))
    inputs = (link_lengths, joint_origins, thetas)

    (ee_x, ee_y, ee_angle), (joint_xs, joint_ys) = lax.scan(scan_fn, init_carry, inputs)

    # Prepend origin (0, 0) to joint positions
    joint_xs = jnp.concatenate([jnp.array([0.0]), joint_xs])
    joint_ys = jnp.concatenate([jnp.array([0.0]), joint_ys])

    return ee_x, ee_y, ee_angle, joint_xs, joint_ys


def _halfspace_point_residual(point_x: Array, point_y: Array, params: Array) -> Array:
    """Compute halfspace residual for a point. Positive = inside."""
    normal_x, normal_y, anchor_x, anchor_y = params[0], params[1], params[2], params[3]
    dx = point_x - anchor_x
    dy = point_y - anchor_y
    return normal_x * dx + normal_y * dy


def _ball_point_residual(point_x: Array, point_y: Array, params: Array) -> Array:
    """Compute ball residual for a point. Positive = inside."""
    center_x, center_y, radius = params[0], params[1], params[2]
    dx = point_x - center_x
    dy = point_y - center_y
    distance = jnp.sqrt(dx**2 + dy**2 + 1e-10)
    return radius - distance


def _rectangle_point_residual(point_x: Array, point_y: Array, params: Array) -> Array:
    """Compute rectangle residual for a point. Positive = inside."""
    left, right, bottom, top = params[0], params[1], params[2], params[3]
    dist_to_left = point_x - left
    dist_to_right = right - point_x
    dist_to_bottom = point_y - bottom
    dist_to_top = top - point_y
    return jnp.minimum(
        jnp.minimum(dist_to_left, dist_to_right),
        jnp.minimum(dist_to_bottom, dist_to_top),
    )


def _compute_nogo_penalty_point(
    joint_xs: Array, joint_ys: Array, nogo_params: NogoRegionParams
) -> Array:
    """Compute nogo penalty using point collision detection."""
    penalty = jnp.float32(0.0)

    # Skip origin (index 0), check all other joint positions
    for i in range(1, joint_xs.shape[0]):
        px, py = joint_xs[i], joint_ys[i]

        # Halfspaces
        def halfspace_penalty(params):
            residual = _halfspace_point_residual(px, py, params)
            return jnp.maximum(residual, 0.0) ** 2

        penalty = penalty + jnp.sum(jax.vmap(halfspace_penalty)(nogo_params.halfspaces))

        # Balls
        def ball_penalty(params):
            residual = _ball_point_residual(px, py, params)
            return jnp.maximum(residual, 0.0) ** 2

        penalty = penalty + jnp.sum(jax.vmap(ball_penalty)(nogo_params.balls))

        # Rectangles
        def rect_penalty(params):
            residual = _rectangle_point_residual(px, py, params)
            return jnp.maximum(residual, 0.0) ** 2

        penalty = penalty + jnp.sum(jax.vmap(rect_penalty)(nogo_params.rectangles))

    return penalty


def _compute_objective(
    thetas: Array,
    robot_params: RobotParams,
    target_x: float,
    target_y: float,
    target_angle: float,
    angle_weight: float,
    nogo_weight: float,
    nogo_params: Optional[NogoRegionParams],
) -> Array:
    """Compute the IK objective function."""
    ee_x, ee_y, ee_angle, joint_xs, joint_ys = _forward_kinematics(
        thetas, robot_params.link_lengths, robot_params.joint_origins
    )

    # Position error
    distance_squared = (ee_x - target_x) ** 2 + (ee_y - target_y) ** 2

    # Angle error with wrapping
    angle_diff = ee_angle - target_angle
    angle_error = jnp.arctan2(jnp.sin(angle_diff), jnp.cos(angle_diff))
    angle_error_squared = angle_error**2

    objective = distance_squared + angle_weight * angle_error_squared

    # Nogo penalty (always computed if nogo_params provided, weight of 0 disables it)
    # Note: nogo_params is None vs not-None is a static condition handled by has_nogo flag
    if nogo_params is not None:
        nogo_penalty = _compute_nogo_penalty_point(joint_xs, joint_ys, nogo_params)
        objective = objective + nogo_weight * nogo_penalty

    return objective


def _optimization_step(
    state: SolverState,
    robot_params: RobotParams,
    target_x: float,
    target_y: float,
    target_angle: float,
    angle_weight: float,
    nogo_weight: float,
    nogo_params: Optional[NogoRegionParams],
    lr: float,
    momentum: float,
    tolerance: float,
) -> SolverState:
    """Single optimization step with momentum."""
    # Compute gradient
    loss, grad = jax.value_and_grad(_compute_objective)(
        state.thetas,
        robot_params,
        target_x,
        target_y,
        target_angle,
        angle_weight,
        nogo_weight,
        nogo_params,
    )

    # Update with momentum
    new_velocity = momentum * state.velocity - lr * grad
    new_thetas = state.thetas + new_velocity

    # Clamp joint angles to limits
    new_thetas = jnp.clip(new_thetas, robot_params.joint_min, robot_params.joint_max)

    # Check convergence
    converged = jnp.abs(state.prev_loss - loss) < tolerance

    return SolverState(
        thetas=new_thetas,
        velocity=new_velocity,
        iteration=state.iteration + 1,
        prev_loss=loss,
        converged=converged,
    )


@partial(jax.jit, static_argnames=["max_iterations", "has_nogo"])
def _solve_ik_jit(
    initial_thetas: Array,
    robot_params: RobotParams,
    target_x: float,
    target_y: float,
    target_angle: float,
    angle_weight: float,
    nogo_weight: float,
    nogo_params: Optional[NogoRegionParams],
    lr: float,
    momentum: float,
    tolerance: float,
    max_iterations: int,
    has_nogo: bool,
) -> SolverResult:
    """JIT-compiled IK solver core.

    Args:
        initial_thetas: Initial joint angles.
        robot_params: Static robot parameters.
        target_x, target_y: Target end effector position.
        target_angle: Target end effector angle.
        angle_weight: Weight for angle constraint.
        nogo_weight: Weight for nogo zone penalty.
        nogo_params: Nogo region parameters (or None).
        lr: Learning rate.
        momentum: Momentum coefficient.
        tolerance: Convergence tolerance.
        max_iterations: Maximum iterations.
        has_nogo: Whether nogo zones are present (static for JIT).

    Returns:
        SolverResult with optimized joint angles.
    """
    # Use nogo_params only if has_nogo is True
    effective_nogo_params = nogo_params if has_nogo else None

    # Clamp initial thetas to joint limits
    initial_thetas = jnp.clip(
        initial_thetas, robot_params.joint_min, robot_params.joint_max
    )

    # Initial state
    init_state = SolverState(
        thetas=initial_thetas,
        velocity=jnp.zeros_like(initial_thetas),
        iteration=0,
        prev_loss=jnp.float32(jnp.inf),
        converged=False,
    )

    # Define loop body
    def loop_body(state: SolverState) -> SolverState:
        return _optimization_step(
            state,
            robot_params,
            target_x,
            target_y,
            target_angle,
            angle_weight,
            nogo_weight,
            effective_nogo_params,
            lr,
            momentum,
            tolerance,
        )

    # Define loop condition
    def loop_cond(state: SolverState) -> bool:
        return jnp.logical_and(
            state.iteration < max_iterations, jnp.logical_not(state.converged)
        )

    # Run optimization loop
    final_state = lax.while_loop(loop_cond, loop_body, init_state)

    return SolverResult(
        thetas=final_state.thetas,
        iterations=final_state.iteration,
        converged=final_state.converged,
        final_loss=final_state.prev_loss,
    )


@dataclass
class IKNumericJAXProfile:
    """Profiling results from IKNumericJAX solver."""

    solve_time_ms: float
    iterations: int
    converged: bool
    initial_loss: float
    final_loss: float
    position_error: float


class IKNumericJAX:
    """Implements a numerical IK solver using JAX with JIT compilation."""

    def __init__(
        self,
        model: RobotModel,
        world: WorldModel | None,
        lr: float = 0.02,
        momentum: float = 0.9,
        max_iterations: int = 500,
        tolerance: float = 1e-6,
        collision_geometry: Literal["line", "point"] = "point",
    ) -> None:
        """Initialize the JAX IK solver.

        Args:
            model: Robot model with link lengths and joint limits.
            world: World model with nogo zones (optional).
            lr: Learning rate for gradient descent.
            momentum: Momentum coefficient (0 = no momentum).
            max_iterations: Maximum optimization iterations.
            tolerance: Convergence tolerance on loss change.
            collision_geometry: Only "point" is supported for JAX solver.
        """
        if collision_geometry != "point":
            raise ValueError(
                "JAX solver only supports point collision geometry for JIT compatibility"
            )

        self.model = model
        self.n_joints = len(model.link_lengths)
        self.collision_geometry = collision_geometry

        # Optimization parameters
        self.lr = lr
        self.momentum = momentum
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Build robot parameters for JIT
        link_lengths = jnp.array(model.link_lengths, dtype=jnp.float32)
        joint_origins = jnp.array(model.joint_origins, dtype=jnp.float32)

        # Process joint limits
        joint_min = []
        joint_max = []
        if model.joint_limits:
            for limit in model.joint_limits:
                if limit is not None:
                    joint_min.append(limit[0])
                    joint_max.append(limit[1])
                else:
                    joint_min.append(-jnp.inf)
                    joint_max.append(jnp.inf)
        # Pad if necessary
        while len(joint_min) < self.n_joints:
            joint_min.append(-jnp.inf)
            joint_max.append(jnp.inf)

        self.robot_params = RobotParams(
            link_lengths=link_lengths,
            joint_origins=joint_origins,
            joint_min=jnp.array(joint_min, dtype=jnp.float32),
            joint_max=jnp.array(joint_max, dtype=jnp.float32),
        )

        # Build nogo parameters
        self.has_nogo = world is not None and len(world.nogo) > 0
        self.nogo_params: Optional[NogoRegionParams] = None

        if self.has_nogo and world is not None:
            halfspaces = []
            balls = []
            rectangles = []

            for region in world.nogo:
                if isinstance(region, RegionHalfspace):
                    halfspaces.append(
                        [
                            region.normal[0],
                            region.normal[1],
                            region.anchor[0],
                            region.anchor[1],
                        ]
                    )
                elif isinstance(region, RegionBall):
                    balls.append([region.center[0], region.center[1], region.radius])
                elif isinstance(region, RegionRectangle):
                    rectangles.append(
                        [region.left, region.right, region.bottom, region.top]
                    )

            # Convert to arrays, using dummy arrays if empty (for JIT compatibility)
            self.nogo_params = NogoRegionParams(
                halfspaces=jnp.array(
                    halfspaces if halfspaces else [[0, 0, 0, 0]], dtype=jnp.float32
                ),
                balls=jnp.array(balls if balls else [[0, 0, 0]], dtype=jnp.float32),
                rectangles=jnp.array(
                    rectangles if rectangles else [[0, 0, 0, 0]], dtype=jnp.float32
                ),
            )

            # Store counts to mask out dummy entries
            self._n_halfspaces = len(halfspaces)
            self._n_balls = len(balls)
            self._n_rectangles = len(rectangles)

    def __call__(
        self,
        state: RobotState,
        desired: DesiredPosition,
        profile: bool = False,
    ) -> RobotState | Tuple[RobotState, IKNumericJAXProfile]:
        """Solve IK for the desired position.

        Args:
            state: Current robot state.
            desired: Desired end effector position and optional angle.
            profile: If True, return profiling information.

        Returns:
            RobotState with solution, optionally with profiling info.
        """
        if state.model != self.model:
            raise ValueError("State model does not match IKNumericJAX model")

        if desired.ee_position is None:
            raise ValueError("DesiredPosition must have an ee_position")

        target_x, target_y = desired.ee_position

        # Extract angle constraint
        if desired.ee_angle is not None:
            target_angle = desired.ee_angle
            angle_weight = 1.0e3
        else:
            target_angle = 0.0
            angle_weight = 0.0

        # Weights
        nogo_weight = 1.0e2 if self.has_nogo else 0.0

        # Initial angles
        initial_thetas = jnp.array(state.current.joint_angles, dtype=jnp.float32)

        # Compute initial loss for profiling
        if profile:
            initial_loss = float(
                _compute_objective(
                    initial_thetas,
                    self.robot_params,
                    target_x,
                    target_y,
                    target_angle,
                    angle_weight,
                    nogo_weight,
                    self.nogo_params if self.has_nogo else None,
                )
            )

        # Start timing
        start_time = time.perf_counter()

        # Run JIT-compiled solver
        result = _solve_ik_jit(
            initial_thetas,
            self.robot_params,
            target_x,
            target_y,
            target_angle,
            angle_weight,
            nogo_weight,
            self.nogo_params,
            self.lr,
            self.momentum,
            self.tolerance,
            self.max_iterations,
            self.has_nogo,
        )

        end_time = time.perf_counter()

        # Extract solution
        joint_angles = tuple(float(a) for a in result.thetas)

        result_state = state.with_position(
            RobotPosition(joint_angles=joint_angles), desired=desired
        )

        if profile:
            # Compute position error
            ee_x, ee_y, _, _, _ = _forward_kinematics(
                result.thetas,
                self.robot_params.link_lengths,
                self.robot_params.joint_origins,
            )
            position_error = float(
                jnp.sqrt((ee_x - target_x) ** 2 + (ee_y - target_y) ** 2)
            )

            profile_result = IKNumericJAXProfile(
                solve_time_ms=(end_time - start_time) * 1000,
                iterations=int(result.iterations),
                converged=bool(result.converged),
                initial_loss=initial_loss,
                final_loss=float(result.final_loss),
                position_error=position_error,
            )
            return result_state, profile_result

        return result_state


if __name__ == "__main__":
    # Interactive IK solver demo using RobotVisualizer
    import math

    from visualization import RobotVisualizer

    # Create a 3-link robot with joint limits
    model = RobotModel(
        link_lengths=(1.0, 0.8, 0.6),
        joint_limits=(
            (0.4 * math.pi, math.pi),
            (-math.pi, 0),
            (-math.pi / 2, math.pi / 2),
        ),
    )

    # Create a world with nogo zones
    nogo = [
        RegionHalfspace((0, -1), (0, -0.2)),
        RegionRectangle(0.5, 10.0, -10.0, 1.0),
        RegionRectangle(0.5, 10.0, 1.6, 5.0),
    ]
    world = WorldModel()

    # Create the IK solver
    ik_solver = IKNumericJAX(
        model,
        world=world,
        collision_geometry="point",
        max_iterations=200,
        lr=0.01,
        momentum=0.9,
    )

    # Initial position (within joint limits)
    initial_position = RobotPosition(joint_angles=(0.5 * math.pi, -math.pi / 4, 0.0))
    current_state = RobotState(model, current=initial_position, world=world)

    # Create visualizer
    viz = RobotVisualizer(current_state)

    # Warm up JIT compilation
    print("Warming up JIT compilation...")
    _ = ik_solver(
        current_state,
        DesiredPosition(ee_position=(1.0, 1.0)),
        profile=True,
    )
    print("JIT compilation complete.")

    # Click callback
    def on_click(x: float, y: float, btn: Literal["left", "right"]):
        global current_state
        print(f"\nClicked at: ({x:.2f}, {y:.2f}) {btn}")

        new_ee_angle: Optional[float] = (
            current_state.desired.ee_angle if current_state.desired else None
        )
        if btn == "right":
            new_ee_angle = 0.0 if new_ee_angle is None else None

        try:
            result = ik_solver(
                current_state,
                DesiredPosition(ee_position=(x, y), ee_angle=new_ee_angle),
                profile=True,
            )
            assert isinstance(result, tuple)
            solution_state, profile = result
            solution = solution_state.current
            print(f"Solution: {tuple(f'{a:.3f}' for a in solution.joint_angles)}")
            print(f"Solve time: {profile.solve_time_ms:.2f}ms")
            print(f"Iterations: {profile.iterations} (converged: {profile.converged})")
            print(f"Loss: {profile.initial_loss:.6f} -> {profile.final_loss:.6f}")
            print(f"Position error: {profile.position_error:.6f}")

            current_state = solution_state
            viz.update(current_state)

        except Exception as e:
            print(f"Error solving IK: {e}")
            import traceback

            traceback.print_exc()

    viz.set_click_callback(on_click)

    print("Interactive IK Solver (JAX)")
    print("=" * 60)
    print("Click anywhere in the window to set a target position.")
    print("The robot will solve IK and move to reach that target.")
    print("=" * 60)

    viz.show()
