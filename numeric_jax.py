#!python3
"""JAX-based numerical IK solver with JIT-compiled core optimization loop."""

import time
from functools import partial
from typing import Literal, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax

from datamodel import (
    DesiredPosition,
    IKReturn,
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


def _halfspace_line_residual(
    p1_x: Array, p1_y: Array, p2_x: Array, p2_y: Array, params: Array
) -> Array:
    """Compute halfspace residual for a line segment. Positive = inside (violation).

    Returns the sum of clamped endpoint residuals (only positive contributions).
    """
    residual1 = _halfspace_point_residual(p1_x, p1_y, params)
    residual2 = _halfspace_point_residual(p2_x, p2_y, params)
    return jnp.maximum(residual1, 0.0) + jnp.maximum(residual2, 0.0)


def _ball_line_residual(
    p1_x: Array, p1_y: Array, p2_x: Array, p2_y: Array, params: Array
) -> Array:
    """Compute ball residual for a line segment. Positive = inside (violation).

    Returns the maximum of endpoint collision and segment collision with the ball.
    """
    center_x, center_y = params[0], params[1]

    # Check endpoint collisions
    residual1 = _ball_point_residual(p1_x, p1_y, params)
    residual2 = _ball_point_residual(p2_x, p2_y, params)
    endpoint_collision = jnp.maximum(residual1, 0.0) + jnp.maximum(residual2, 0.0)

    # Vector from p1 to p2
    seg_dx = p2_x - p1_x
    seg_dy = p2_y - p1_y

    # Vector from p1 to center
    to_center_x = center_x - p1_x
    to_center_y = center_y - p1_y

    # Project center onto line, clamped to segment [0, 1]
    seg_len_sq = seg_dx**2 + seg_dy**2 + 1e-10  # Avoid division by zero
    t = (to_center_x * seg_dx + to_center_y * seg_dy) / seg_len_sq
    t = jnp.clip(t, 0.0, 1.0)

    # Closest point on segment
    closest_x = p1_x + t * seg_dx
    closest_y = p1_y + t * seg_dy

    # Residual at closest point on segment
    segment_collision = jnp.maximum(
        _ball_point_residual(closest_x, closest_y, params), 0.0
    )

    return jnp.maximum(endpoint_collision, segment_collision)


def _rectangle_line_residual(
    p1_x: Array, p1_y: Array, p2_x: Array, p2_y: Array, params: Array
) -> Array:
    """Compute rectangle residual for a line segment. Positive = inside (violation).

    Returns the maximum of endpoint collision and edge intersection collision.
    """
    left, right, bottom, top = params[0], params[1], params[2], params[3]

    # Check endpoint collisions
    residual1 = _rectangle_point_residual(p1_x, p1_y, params)
    residual2 = _rectangle_point_residual(p2_x, p2_y, params)
    endpoint_collision = jnp.maximum(residual1, 0.0) + jnp.maximum(residual2, 0.0)

    # Vector from p1 to p2
    dx = p2_x - p1_x
    dy = p2_y - p1_y

    # Helper to compute edge intersection score
    def edge_intersection_score(
        t: Array, edge_coord: Array, edge_min: Array, edge_max: Array
    ) -> Array:
        # Check t in [0, 1]
        t_valid = jnp.minimum(t, 1.0 - t)
        # Check coord in [edge_min, edge_max]
        coord_valid = jnp.minimum(edge_coord - edge_min, edge_max - edge_coord)
        # Both must be non-negative
        return jnp.maximum(jnp.minimum(t_valid, coord_valid), 0.0)

    # Left edge intersection
    t_left = (left - p1_x) / (dx + 1e-10)
    y_at_left = p1_y + t_left * dy
    left_collision = edge_intersection_score(t_left, y_at_left, bottom, top)

    # Right edge intersection
    t_right = (right - p1_x) / (dx + 1e-10)
    y_at_right = p1_y + t_right * dy
    right_collision = edge_intersection_score(t_right, y_at_right, bottom, top)

    # Bottom edge intersection
    t_bottom = (bottom - p1_y) / (dy + 1e-10)
    x_at_bottom = p1_x + t_bottom * dx
    bottom_collision = edge_intersection_score(t_bottom, x_at_bottom, left, right)

    # Top edge intersection
    t_top = (top - p1_y) / (dy + 1e-10)
    x_at_top = p1_x + t_top * dx
    top_collision = edge_intersection_score(t_top, x_at_top, left, right)

    edge_collision = left_collision + right_collision + bottom_collision + top_collision

    return jnp.maximum(endpoint_collision, edge_collision)


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


def _compute_nogo_penalty_line(
    joint_xs: Array, joint_ys: Array, nogo_params: NogoRegionParams
) -> Array:
    """Compute nogo penalty using line segment collision detection."""
    penalty = jnp.float32(0.0)

    # Check each link segment (from joint i to joint i+1)
    for i in range(joint_xs.shape[0] - 1):
        p1_x, p1_y = joint_xs[i], joint_ys[i]
        p2_x, p2_y = joint_xs[i + 1], joint_ys[i + 1]

        # Halfspaces
        def halfspace_penalty(params):
            residual = _halfspace_line_residual(p1_x, p1_y, p2_x, p2_y, params)
            return residual**2

        penalty = penalty + jnp.sum(jax.vmap(halfspace_penalty)(nogo_params.halfspaces))

        # Balls
        def ball_penalty(params):
            residual = _ball_line_residual(p1_x, p1_y, p2_x, p2_y, params)
            return residual**2

        penalty = penalty + jnp.sum(jax.vmap(ball_penalty)(nogo_params.balls))

        # Rectangles
        def rect_penalty(params):
            residual = _rectangle_line_residual(p1_x, p1_y, p2_x, p2_y, params)
            return residual**2

        penalty = penalty + jnp.sum(jax.vmap(rect_penalty)(nogo_params.rectangles))

    return penalty


def _compute_objective(
    thetas: Array,
    robot_params: RobotParams,
    target_x: float,
    target_y: float,
    nogo_weight: float,
    nogo_params: Optional[NogoRegionParams],
    use_line_collision: bool,
) -> Array:
    """Compute the IK objective function."""
    ee_x, ee_y, _, joint_xs, joint_ys = _forward_kinematics(
        thetas, robot_params.link_lengths, robot_params.joint_origins
    )

    # Position error
    objective = (ee_x - target_x) ** 2 + (ee_y - target_y) ** 2

    # Nogo penalty (always computed if nogo_params provided, weight of 0 disables it)
    # Note: nogo_params is None vs not-None is a static condition handled by has_nogo flag
    if nogo_params is not None:
        nogo_penalty = lax.cond(
            use_line_collision,
            lambda: _compute_nogo_penalty_line(joint_xs, joint_ys, nogo_params),
            lambda: _compute_nogo_penalty_point(joint_xs, joint_ys, nogo_params),
        )
        objective = objective + nogo_weight * nogo_penalty

    return objective


def _set_last_joint_for_angle(
    thetas: Array,
    robot_params: RobotParams,
    target_angle: float,
) -> Array:
    """Set the last joint angle to achieve the desired end effector angle.

    The end effector angle is the sum of all joint angles plus joint origins.
    To achieve a target angle, we set the last joint to:
        theta_last = target_angle - sum(theta_i + origin_i for i in 0..n-2) - origin_last
    """
    n_joints = thetas.shape[0]

    # Compute cumulative angle from all joints except the last
    cumulative_angle = jnp.float32(0.0)
    for i in range(n_joints - 1):
        cumulative_angle = cumulative_angle + thetas[i] + robot_params.joint_origins[i]

    # Required last joint angle to achieve target
    required_last = (
        target_angle - cumulative_angle - robot_params.joint_origins[n_joints - 1]
    )

    # Normalize to [-pi, pi]
    required_last = jnp.arctan2(jnp.sin(required_last), jnp.cos(required_last))

    # Clamp to joint limits
    required_last = jnp.clip(
        required_last,
        robot_params.joint_min[n_joints - 1],
        robot_params.joint_max[n_joints - 1],
    )

    # Update thetas with new last joint angle
    return thetas.at[n_joints - 1].set(required_last)


def _optimization_step(
    state: SolverState,
    robot_params: RobotParams,
    target_x: float,
    target_y: float,
    target_angle: float,
    lock_angle: bool,
    nogo_weight: float,
    nogo_params: Optional[NogoRegionParams],
    use_line_collision: bool,
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
        nogo_weight,
        nogo_params,
        use_line_collision,
    )

    # Update with momentum
    new_velocity = momentum * state.velocity - lr * grad
    new_thetas = state.thetas + new_velocity

    # Clamp joint angles to limits
    new_thetas = jnp.clip(new_thetas, robot_params.joint_min, robot_params.joint_max)

    # If angle locking is enabled, set the last joint to achieve the target angle
    new_thetas = lax.cond(
        lock_angle,
        lambda t: _set_last_joint_for_angle(t, robot_params, target_angle),
        lambda t: t,
        new_thetas,
    )

    # Check convergence
    converged = jnp.abs(state.prev_loss - loss) < tolerance

    return SolverState(
        thetas=new_thetas,
        velocity=new_velocity,
        iteration=state.iteration + 1,
        prev_loss=loss,
        converged=converged,  # type: ignore
    )


@partial(jax.jit, static_argnames=["max_iterations", "has_nogo", "use_line_collision"])
def _solve_ik_jit(
    initial_thetas: Array,
    robot_params: RobotParams,
    target_x: float,
    target_y: float,
    target_angle: float,
    lock_angle: bool,
    nogo_weight: float,
    nogo_params: Optional[NogoRegionParams],
    use_line_collision: bool,
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
        target_angle: Target end effector angle (used if lock_angle is True).
        lock_angle: Whether to lock the end effector angle.
        nogo_weight: Weight for nogo zone penalty.
        nogo_params: Nogo region parameters (or None).
        use_line_collision: Whether to use line collision (True) or point collision (False).
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

    # If angle locking is enabled, set the last joint initially
    initial_thetas = lax.cond(
        lock_angle,
        lambda t: _set_last_joint_for_angle(t, robot_params, target_angle),
        lambda t: t,
        initial_thetas,
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
            lock_angle,
            nogo_weight,
            effective_nogo_params,
            use_line_collision,
            lr,
            momentum,
            tolerance,
        )

    # Define loop condition
    def loop_cond(state: SolverState) -> bool:
        return jnp.logical_and(
            state.iteration < max_iterations, jnp.logical_not(state.converged)
        )  # type: ignore

    # Run optimization loop
    final_state = lax.while_loop(loop_cond, loop_body, init_state)

    return SolverResult(
        thetas=final_state.thetas,
        iterations=final_state.iteration,
        converged=final_state.converged,
        final_loss=final_state.prev_loss,
    )


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
            collision_geometry: "point" for joint-only collision, "line" for link segments.
        """
        self.model = model
        self.n_joints = len(model.link_lengths)
        self.collision_geometry = collision_geometry
        self.use_line_collision = collision_geometry == "line"

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
    ) -> IKReturn:
        """Solve IK for the desired position.

        Args:
            state: Current robot state.
            desired: Desired end effector position and optional angle.

        Returns:
            IKReturn containing the solution state and profiling information.
        """
        if state.model != self.model:
            raise ValueError("State model does not match IKNumericJAX model")

        if desired.ee_position is None:
            raise ValueError("DesiredPosition must have an ee_position")

        target_x, target_y = desired.ee_position

        # Extract angle constraint
        lock_angle = desired.ee_angle is not None
        target_angle = desired.ee_angle if lock_angle else 0.0

        # Weights
        nogo_weight = 0.0
        if self.has_nogo:
            nogo_weight = 2.0 if self.use_line_collision else 20.0

        # Initial angles
        initial_thetas = jnp.array(state.current.joint_angles, dtype=jnp.float32)

        # Compute initial loss
        initial_loss = float(
            _compute_objective(
                initial_thetas,
                self.robot_params,
                target_x,
                target_y,
                nogo_weight,
                self.nogo_params if self.has_nogo else None,
                self.use_line_collision,
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
            lock_angle,
            nogo_weight,
            self.nogo_params,
            self.use_line_collision,
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

        # Compute position error
        ee_x, ee_y, _, _, _ = _forward_kinematics(
            result.thetas,
            self.robot_params.link_lengths,
            self.robot_params.joint_origins,
        )
        position_error = float(
            jnp.sqrt((ee_x - target_x) ** 2 + (ee_y - target_y) ** 2)
        )

        return IKReturn(
            state=result_state,
            solve_time_ms=(end_time - start_time) * 1000,
            iterations=int(result.iterations),
            converged=bool(result.converged),
            initial_loss=initial_loss,
            final_loss=float(result.final_loss),
            position_error=position_error,
        )
