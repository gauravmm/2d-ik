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
    IKSolver,
    Region,
    RegionBall,
    RegionHalfspace,
    RegionRectangle,
    RobotModel,
    RobotPosition,
    RobotState,
    WorldModel,
)

# Register dataclasses as JAX PyTree nodes
jax.tree_util.register_dataclass(
    RegionHalfspace,
    data_fields=["normal", "anchor"],
    meta_fields=[],
)
jax.tree_util.register_dataclass(
    RegionBall,
    data_fields=["center", "radius"],
    meta_fields=[],
)
jax.tree_util.register_dataclass(
    RegionRectangle,
    data_fields=["left", "right", "bottom", "top"],
    meta_fields=[],
)
jax.tree_util.register_dataclass(
    RobotModel,
    data_fields=[],
    meta_fields=["link_lengths", "joint_origins", "joint_limits"],
)

NOGO_PENALTY_LINE = 2.0
NOGO_PENALTY_POINT = 20.0

# Type alias for JAX arrays
Array = jax.Array


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


def _halfspace_point_residual(
    point_x: Array, point_y: Array, region: RegionHalfspace
) -> Array:
    """Compute halfspace residual for a point. Positive = inside."""
    dx = point_x - region.anchor[0]
    dy = point_y - region.anchor[1]
    return region.normal[0] * dx + region.normal[1] * dy


def _ball_point_residual(point_x: Array, point_y: Array, region: RegionBall) -> Array:
    """Compute ball residual for a point. Positive = inside."""
    dx = point_x - region.center[0]
    dy = point_y - region.center[1]
    distance = jnp.sqrt(dx**2 + dy**2 + 1e-10)
    return region.radius - distance


def _rectangle_point_residual(
    point_x: Array, point_y: Array, region: RegionRectangle
) -> Array:
    """Compute rectangle residual for a point. Positive = inside."""
    dist_to_left = point_x - region.left
    dist_to_right = region.right - point_x
    dist_to_bottom = point_y - region.bottom
    dist_to_top = region.top - point_y
    return jnp.minimum(
        jnp.minimum(dist_to_left, dist_to_right),
        jnp.minimum(dist_to_bottom, dist_to_top),
    )


def _halfspace_line_residual(
    p1_x: Array, p1_y: Array, p2_x: Array, p2_y: Array, region: RegionHalfspace
) -> Array:
    """Compute halfspace residual for a line segment. Positive = inside (violation).

    Returns the sum of clamped endpoint residuals (only positive contributions).
    """
    residual1 = _halfspace_point_residual(p1_x, p1_y, region)
    residual2 = _halfspace_point_residual(p2_x, p2_y, region)
    return jnp.maximum(residual1, 0.0) + jnp.maximum(residual2, 0.0)


def _ball_line_residual(
    p1_x: Array, p1_y: Array, p2_x: Array, p2_y: Array, region: RegionBall
) -> Array:
    """Compute ball residual for a line segment. Positive = inside (violation).

    Returns the maximum of endpoint collision and segment collision with the ball.
    """
    center_x, center_y = region.center[0], region.center[1]

    # Check endpoint collisions
    residual1 = _ball_point_residual(p1_x, p1_y, region)
    residual2 = _ball_point_residual(p2_x, p2_y, region)
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
        _ball_point_residual(closest_x, closest_y, region), 0.0
    )

    return jnp.maximum(endpoint_collision, segment_collision)


def _rectangle_line_residual(
    p1_x: Array, p1_y: Array, p2_x: Array, p2_y: Array, region: RegionRectangle
) -> Array:
    """Compute rectangle residual for a line segment. Positive = inside (violation).

    Returns the maximum of endpoint collision and edge intersection collision.
    """
    left, right, bottom, top = region.left, region.right, region.bottom, region.top

    # Check endpoint collisions
    residual1 = _rectangle_point_residual(p1_x, p1_y, region)
    residual2 = _rectangle_point_residual(p2_x, p2_y, region)
    endpoint_collision = jnp.maximum(residual1, 0.0) + jnp.maximum(residual2, 0.0)

    # Vector from p1 to p2
    dx = p2_x - p1_x
    dy = p2_y - p1_y

    # Helper to compute edge intersection score
    def edge_intersection_score(
        t: Array, edge_coord: Array, edge_min: float, edge_max: float
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
    joint_xs: Array, joint_ys: Array, nogo_regions: Tuple[Region, ...] | None
) -> Array:
    """Compute nogo penalty using point collision detection."""
    penalty = jnp.float32(0.0)
    if nogo_regions is None:
        return penalty

    # Skip origin (index 0), check all other joint positions
    for i in range(1, joint_xs.shape[0]):
        px, py = joint_xs[i], joint_ys[i]

        # Halfspaces
        for region in nogo_regions:
            if isinstance(region, RegionHalfspace):
                residual = _halfspace_point_residual(px, py, region)
            elif isinstance(region, RegionBall):
                residual = _ball_point_residual(px, py, region)
            elif isinstance(region, RegionRectangle):
                residual = _rectangle_point_residual(px, py, region)
            else:
                residual = 0.0

            penalty = penalty + jnp.maximum(residual, 0.0) ** 2

    return penalty


def _compute_nogo_penalty_line(
    joint_xs: Array, joint_ys: Array, nogo_regions: Tuple[Region, ...] | None
) -> Array:
    """Compute nogo penalty using line segment collision detection."""
    penalty = jnp.float32(0.0)
    if nogo_regions is None:
        return penalty

    # Check each link segment (from joint i to joint i+1)
    for i in range(joint_xs.shape[0] - 1):
        p1_x, p1_y = joint_xs[i], joint_ys[i]
        p2_x, p2_y = joint_xs[i + 1], joint_ys[i + 1]

        for region in nogo_regions:
            if isinstance(region, RegionHalfspace):
                residual = _halfspace_line_residual(p1_x, p1_y, p2_x, p2_y, region)
            elif isinstance(region, RegionBall):
                residual = _ball_line_residual(p1_x, p1_y, p2_x, p2_y, region)
            elif isinstance(region, RegionRectangle):
                residual = _rectangle_line_residual(p1_x, p1_y, p2_x, p2_y, region)
            else:
                residual = 0.0

            penalty = penalty + residual**2

    return penalty


def _compute_objective(
    thetas: Array,
    model: RobotModel,
    target_x: float,
    target_y: float,
    nogo_regions: Optional[Tuple[Region, ...]],  # Static
    use_line_collision: bool,  # Static
) -> Array:
    """Compute the IK objective function."""
    link_lengths = jnp.array(model.link_lengths, dtype=jnp.float32)
    joint_origins = jnp.array(model.joint_origins, dtype=jnp.float32)
    ee_x, ee_y, _, joint_xs, joint_ys = _forward_kinematics(
        thetas, link_lengths, joint_origins
    )

    # Position error
    objective = (ee_x - target_x) ** 2 + (ee_y - target_y) ** 2

    # Note: nogo_regions is None vs not-None is a static condition. Any compilation must
    # specify nogo_regions as a static argument.
    if nogo_regions is not None:
        if use_line_collision:
            nogo_penalty = NOGO_PENALTY_LINE * _compute_nogo_penalty_line(
                joint_xs, joint_ys, nogo_regions
            )
        else:
            nogo_penalty = NOGO_PENALTY_POINT * _compute_nogo_penalty_point(
                joint_xs, joint_ys, nogo_regions
            )
        objective = objective + nogo_penalty

    return objective


def _set_last_joint_for_angle(
    thetas: Array,
    model: RobotModel,
    target_angle: float,
) -> Array:
    """Set the last joint angle to achieve the desired end effector angle.

    The end effector angle is the sum of all joint angles plus joint origins.
    To achieve a target angle, we set the last joint to:
        theta_last = target_angle - sum(theta_i + origin_i for i in 0..n-2) - origin_last
    """
    n_joints = thetas.shape[0]
    joint_origins = jnp.array(model.joint_origins, dtype=jnp.float32)
    joint_limits = model.joint_limits

    # Compute cumulative angle from all joints except the last
    cumulative_angle = jnp.float32(0.0)
    for i in range(n_joints - 1):
        cumulative_angle = cumulative_angle + thetas[i] + joint_origins[i]

    # Required last joint angle to achieve target
    required_last = target_angle - cumulative_angle - joint_origins[n_joints - 1]

    # Normalize to [-pi, pi]
    required_last = jnp.arctan2(jnp.sin(required_last), jnp.cos(required_last))

    # Clamp to joint limits
    required_last = jnp.clip(
        required_last,
        joint_limits[n_joints - 1][0],
        joint_limits[n_joints - 1][1],
    )

    # Update thetas with new last joint angle
    return thetas.at[n_joints - 1].set(required_last)


def _optimization_step(
    state: SolverState,
    model: RobotModel,
    target_x: float,
    target_y: float,
    target_angle: float,
    lock_angle: bool,
    nogo_regions: Optional[Tuple[Region, ...]],
    use_line_collision: bool,
    lr: float,
    momentum: float,
    tolerance: float,
) -> SolverState:
    """Single optimization step with momentum."""
    # Compute gradient
    loss, grad = jax.value_and_grad(_compute_objective)(
        state.thetas,
        model,
        target_x,
        target_y,
        nogo_regions,
        use_line_collision,
    )

    # Update with momentum
    new_velocity = momentum * state.velocity - lr * grad
    new_thetas = state.thetas + new_velocity

    # Clamp joint angles to limits
    joint_min = jnp.array([lim[0] for lim in model.joint_limits], dtype=jnp.float32)
    joint_max = jnp.array([lim[1] for lim in model.joint_limits], dtype=jnp.float32)
    new_thetas = jnp.clip(new_thetas, joint_min, joint_max)

    # If angle locking is enabled, set the last joint to achieve the target angle
    new_thetas = lax.cond(
        lock_angle,
        lambda t: _set_last_joint_for_angle(t, model, target_angle),
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


@partial(
    jax.jit,
    static_argnames=[
        "max_iterations",
        "model",
        "use_line_collision",
        "nogo_regions",
    ],
)
def _solve_ik_jit(
    initial_thetas: Array,
    model: RobotModel,
    target_x: float,
    target_y: float,
    target_angle: float,
    lock_angle: bool,
    nogo_regions: Optional[Tuple[Region, ...]],
    use_line_collision: bool,
    lr: float,
    momentum: float,
    tolerance: float,
    max_iterations: int,
) -> SolverResult:
    """JIT-compiled IK solver core.

    Args:
        initial_thetas: Initial joint angles.
        model: Static robot model.
        target_x, target_y: Target end effector position.
        target_angle: Target end effector angle (used if lock_angle is True).
        lock_angle: Whether to lock the end effector angle.
        nogo_regions: Nogo regions (or None).
        use_line_collision: Whether to use line collision (True) or point collision (False).
        lr: Learning rate.
        momentum: Momentum coefficient.
        tolerance: Convergence tolerance.
        max_iterations: Maximum iterations.

    Returns:
        SolverResult with optimized joint angles.
    """
    # Clamp initial thetas to joint limits
    joint_min = jnp.array([lim[0] for lim in model.joint_limits], dtype=jnp.float32)
    joint_max = jnp.array([lim[1] for lim in model.joint_limits], dtype=jnp.float32)
    initial_thetas = jnp.clip(initial_thetas, joint_min, joint_max)

    # If angle locking is enabled, set the last joint initially
    initial_thetas = lax.cond(
        lock_angle,
        lambda t: _set_last_joint_for_angle(t, model, target_angle),
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
            model,
            target_x,
            target_y,
            target_angle,
            lock_angle,
            nogo_regions,
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


class IKNumericJAX(IKSolver):
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

    def starting(self, state: RobotState) -> None:
        """Warm up the JIT-compiled solver by running a dummy solve.

        This triggers JAX JIT compilation so subsequent calls are fast.

        Args:
            state: The initial robot state.
        """
        if state.model != self.model:
            raise ValueError("State model does not match IKNumericJAX model")

        # Run a dummy solve to trigger JIT compilation
        initial_thetas = jnp.array(state.current.joint_angles, dtype=jnp.float32)
        # Use current end effector position as target for warmup
        positions = state.get_joint_positions()
        target_x, target_y = positions[-1]

        _solve_ik_jit(
            initial_thetas,
            self.model,
            target_x,
            target_y,
            0.0,  # target_angle
            False,  # lock_angle
            state.world.nogo,
            self.use_line_collision,
            self.lr,
            self.momentum,
            self.tolerance,
            self.max_iterations,
        )

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

        # Initial angles
        initial_thetas = jnp.array(state.current.joint_angles, dtype=jnp.float32)

        # Start timing
        start_time = time.perf_counter()

        # Run JIT-compiled solver
        result = _solve_ik_jit(
            initial_thetas,
            self.model,
            target_x,
            target_y,
            target_angle,
            lock_angle,
            state.world.nogo,
            self.use_line_collision,
            self.lr,
            self.momentum,
            self.tolerance,
            self.max_iterations,
        )

        end_time = time.perf_counter()

        # Extract solution
        joint_angles = tuple(float(a) for a in result.thetas)

        result_state = state.with_position(
            RobotPosition(joint_angles=joint_angles), desired=desired
        )

        # Compute position error
        link_lengths = jnp.array(self.model.link_lengths, dtype=jnp.float32)
        joint_origins = jnp.array(self.model.joint_origins, dtype=jnp.float32)
        ee_x, ee_y, _, _, _ = _forward_kinematics(
            result.thetas,
            link_lengths,
            joint_origins,
        )
        position_error = float(
            jnp.sqrt((ee_x - target_x) ** 2 + (ee_y - target_y) ** 2)
        )

        return IKReturn(
            state=result_state,
            solve_time_ms=(end_time - start_time) * 1000,
            iterations=int(result.iterations),
            converged=bool(result.converged),
            initial_loss=-1,
            final_loss=float(result.final_loss),
            position_error=position_error,
        )
