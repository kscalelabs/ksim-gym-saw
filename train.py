"""Defines simple task for training a walking policy for the default humanoid."""

import asyncio
import functools
import math
from dataclasses import dataclass
from typing import Collection, Self

import attrs
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import ksim
import mujoco
import mujoco_scenes
import mujoco_scenes.mjcf
import optax
import xax
from jaxtyping import Array, PRNGKeyArray

# These are in the order of the neural network outputs.
ZEROS: list[tuple[str, float]] = [
    ("dof_right_shoulder_pitch_03", 0.0),
    ("dof_right_shoulder_roll_03", math.radians(-10.0)),
    ("dof_right_shoulder_yaw_02", 0.0),
    ("dof_right_elbow_02", math.radians(90.0)),
    ("dof_right_wrist_00", 0.0),
    ("dof_left_shoulder_pitch_03", 0.0),
    ("dof_left_shoulder_roll_03", math.radians(10.0)),
    ("dof_left_shoulder_yaw_02", 0.0),
    ("dof_left_elbow_02", math.radians(-90.0)),
    ("dof_left_wrist_00", 0.0),
    ("dof_right_hip_pitch_04", math.radians(-20.0)),
    ("dof_right_hip_roll_03", math.radians(-0.0)),
    ("dof_right_hip_yaw_03", 0.0),
    ("dof_right_knee_04", math.radians(-50.0)),
    ("dof_right_ankle_02", math.radians(30.0)),
    ("dof_left_hip_pitch_04", math.radians(20.0)),
    ("dof_left_hip_roll_03", math.radians(0.0)),
    ("dof_left_hip_yaw_03", 0.0),
    ("dof_left_knee_04", math.radians(50.0)),
    ("dof_left_ankle_02", math.radians(-30.0)),
]


@dataclass
class HumanoidWalkingTaskConfig(ksim.PPOConfig):
    """Config for the humanoid walking task."""

    # Model parameters.
    hidden_size: int = xax.field(
        value=128,
        help="The hidden size for the MLPs.",
    )
    depth: int = xax.field(
        value=5,
        help="The depth for the MLPs.",
    )
    num_mixtures: int = xax.field(
        value=5,
        help="The number of mixtures for the actor.",
    )
    var_scale: float = xax.field(
        value=0.5,
        help="The scale for the standard deviations of the actor.",
    )
    use_acc_gyro: bool = xax.field(
        value=True,
        help="Whether to use the IMU acceleration and gyroscope observations.",
    )

    # Optimizer parameters.
    learning_rate: float = xax.field(
        value=3e-4,
        help="Learning rate for PPO.",
    )
    adam_weight_decay: float = xax.field(
        value=1e-5,
        help="Weight decay for the Adam optimizer.",
    )


@attrs.define(kw_only=True)
class VelocityCommandMarker(ksim.vis.Marker):
    """Visualize velocity commands with different colors and shapes for each movement category.

    LEGEND:
    - Stand still: Gray circle (sphere)
    - Sagittal (forward/backward): Green arrow pointing forward/backward relative to robot
    - Lateral (sideways): Blue arrow pointing left/right relative to robot
    - Rotate: Red arrow pointing up/down (clockwise/counterclockwise rotation)
    - Omni (combined movement): Black arrow in the direction of combined velocity
    """

    command_name: str = attrs.field()
    radius: float = attrs.field(default=0.08)
    size: float = attrs.field(default=0.04)
    arrow_len: float = attrs.field(default=0.3)
    height: float = attrs.field(default=0.6)

    def update(self, trajectory: ksim.Trajectory) -> None:
        cmd = trajectory.command[self.command_name]  # [cx, cy, cyaw, steps_left]
        cx, cy, cyaw = float(cmd[0]), float(cmd[1]), float(cmd[2])

        self.pos = (0.0, 0.0, self.height)

        # Determine movement category based on which components are non-zero
        has_x = abs(cx) > 1e-4
        has_y = abs(cy) > 1e-4
        has_yaw = abs(cyaw) > 1e-4

        if not (has_x or has_y or has_yaw):
            # Stand still: Gray circle
            self.geom = mujoco.mjtGeom.mjGEOM_SPHERE
            self.scale = (self.radius, self.radius, self.radius)
            self.rgba = (0.7, 0.7, 0.7, 0.8)

        elif has_x and not has_y and not has_yaw:
            # Sagittal (forward/backward): Green arrow
            self.geom = mujoco.mjtGeom.mjGEOM_ARROW
            direction = (1.0 if cx > 0 else -1.0, 0.0, 0.0)
            self.scale = (self.size, self.size, self.arrow_len * abs(cx))
            self.orientation = self.quat_from_direction(direction)
            self.rgba = (0.2, 0.8, 0.2, 0.8)  # Green

        elif has_y and not has_x and not has_yaw:
            # Lateral (sideways): Blue arrow
            self.geom = mujoco.mjtGeom.mjGEOM_ARROW
            direction = (0.0, 1.0 if cy > 0 else -1.0, 0.0)
            self.scale = (self.size, self.size, self.arrow_len * abs(cy))
            self.orientation = self.quat_from_direction(direction)
            self.rgba = (0.2, 0.2, 0.8, 0.8)  # Blue

        elif has_yaw and not has_x and not has_y:
            # Rotate: Red arrow pointing up/down
            self.geom = mujoco.mjtGeom.mjGEOM_ARROW
            direction = (0.0, 0.0, 1.0 if cyaw > 0 else -1.0)
            self.scale = (self.size, self.size, self.arrow_len * abs(cyaw))
            self.orientation = self.quat_from_direction(direction)
            self.rgba = (0.8, 0.2, 0.2, 0.8)  # Red

        else:
            # Omni (combined movement): Black arrow in direction of combined velocity
            self.geom = mujoco.mjtGeom.mjGEOM_ARROW
            # For combined movement, use the linear velocity direction
            if has_x or has_y:
                magnitude = math.sqrt(cx * cx + cy * cy)
                direction = (cx / magnitude if magnitude > 0 else 1.0, cy / magnitude if magnitude > 0 else 0.0, 0.0)
                self.scale = (self.size, self.size, self.arrow_len * magnitude)
            else:
                # Only rotation, use up/down
                direction = (0.0, 0.0, 1.0 if cyaw > 0 else -1.0)
                self.scale = (self.size, self.size, self.arrow_len * abs(cyaw))

            self.orientation = self.quat_from_direction(direction)
            self.rgba = (0.1, 0.1, 0.1, 0.8)  # Black

    @classmethod
    def get(cls, command_name: str, *, height: float = 0.6) -> Self:
        return cls(
            command_name=command_name,
            target_type="root",
            geom=mujoco.mjtGeom.mjGEOM_SPHERE,
            scale=(0.0, 0.0, 0.0),
            height=height,
            track_rotation=True,
        )


@attrs.define(frozen=True, kw_only=True)
class VelocityCommand(ksim.Command):
    """Body-frame velocity command with built-in resampling timer."""

    categories: tuple[str, ...] = ("stand", "sagittal", "lateral", "rotate", "omni")
    NUM_CATS: int = len(categories)
    # sampling ranges
    x_range: tuple[float, float] = (-0.5, 2.0)
    y_range: tuple[float, float] = (-0.5, 0.5)
    yaw_range: tuple[float, float] = (-0.5, 0.5)

    # timing
    interval_range: tuple[float, float] = (2.0, 6.0)
    dt: float = attrs.field()

    @staticmethod
    def to_policy(cmd: jnp.ndarray) -> jnp.ndarray:  # [cx,cy,cyaw]
        return cmd[..., :3]

    def initial_command(
        self,
        _physics_data: ksim.PhysicsData,
        _curriculum_level: Array,
        _rng: PRNGKeyArray,
    ) -> jnp.ndarray:
        """Start with zeros so the first step triggers a resample."""
        return jnp.zeros(4, dtype=jnp.float32)

    def _sample_category(self, rng: PRNGKeyArray) -> jnp.ndarray:
        return jax.random.randint(rng, (), 0, self.NUM_CATS, dtype=jnp.int32)

    def _sample_cmd(self, cat: jnp.ndarray, rng: PRNGKeyArray) -> jnp.ndarray:
        """Sample ``[cx, cy, cyaw]`` according to the motion category."""
        rng_x, rng_y, rng_yaw = jax.random.split(rng, 3)

        # Draw one value for each axis (they may or may not be used).
        cx = jax.random.uniform(rng_x, (), minval=self.x_range[0], maxval=self.x_range[1])
        cy = jax.random.uniform(rng_y, (), minval=self.y_range[0], maxval=self.y_range[1])
        wz = jax.random.uniform(rng_yaw, (), minval=self.yaw_range[0], maxval=self.yaw_range[1])

        # Command table: shape (5, 3)
        cmd_table = jnp.stack(
            [
                jnp.array([0.0, 0.0, 0.0]),  # stand
                jnp.array([cx, 0.0, 0.0]),  # sagittal walk
                jnp.array([0.0, cy, 0.0]),  # lateral walk
                jnp.array([0.0, 0.0, wz]),  # rotate in place
                jnp.array([cx, cy, wz]),  # omnidirectional
            ]
        )

        # `cat` is the integer index selecting the row we want.
        return cmd_table[cat]

    def __call__(
        self,
        prev_command: jnp.ndarray,  # (4,)
        _physics_data: ksim.PhysicsData,
        _curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> jnp.ndarray:
        cmd_prev, steps_left = prev_command[:3], prev_command[3] - 1.0

        def resample(_: None) -> jnp.ndarray:
            rng_cat, rng_cmd, rng_int = jax.random.split(rng, 3)
            cat = self._sample_category(rng_cat)
            cmd_vec = self._sample_cmd(cat, rng_cmd)  # (3,)
            secs = jax.random.uniform(rng_int, (), minval=self.interval_range[0], maxval=self.interval_range[1])
            new_steps = jnp.maximum(jnp.round(secs / self.dt), 1.0)
            return jnp.concatenate([cmd_vec, new_steps[None]], axis=0)

        def keep(_: None) -> jnp.ndarray:
            return jnp.concatenate([cmd_prev, steps_left[None]], axis=0)

        return jax.lax.cond(steps_left <= 0.0, resample, keep, operand=None)

    def get_markers(self) -> Collection[ksim.vis.Marker]:
        """Get the visualization markers for this command."""
        return [VelocityCommandMarker.get(self.command_name)]


@attrs.define(frozen=True, kw_only=True)
class JointPositionPenalty(ksim.JointDeviationPenalty):
    @classmethod
    def create_from_names(
        cls,
        names: list[str],
        physics_model: ksim.PhysicsModel,
        scale: float = -1.0,
        scale_by_curriculum: bool = False,
    ) -> Self:
        zeros = {k: v for k, v in ZEROS}
        joint_targets = [zeros[name] for name in names]

        return cls.create(
            physics_model=physics_model,
            joint_names=tuple(names),
            joint_targets=tuple(joint_targets),
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


@attrs.define(frozen=True, kw_only=True)
class BentArmPenalty(JointPositionPenalty):
    @classmethod
    def create_penalty(
        cls,
        physics_model: ksim.PhysicsModel,
        scale: float = -1.0,
        scale_by_curriculum: bool = False,
    ) -> Self:
        return cls.create_from_names(
            names=[
                "dof_right_shoulder_pitch_03",
                "dof_right_shoulder_roll_03",
                "dof_right_shoulder_yaw_02",
                "dof_right_elbow_02",
                "dof_right_wrist_00",
                "dof_left_shoulder_pitch_03",
                "dof_left_shoulder_roll_03",
                "dof_left_shoulder_yaw_02",
                "dof_left_elbow_02",
                "dof_left_wrist_00",
            ],
            physics_model=physics_model,
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


@attrs.define(frozen=True, kw_only=True)
class StraightLegPenalty(JointPositionPenalty):
    @classmethod
    def create_penalty(
        cls,
        physics_model: ksim.PhysicsModel,
        scale: float = -1.0,
        scale_by_curriculum: bool = False,
    ) -> Self:
        return cls.create_from_names(
            names=[
                "dof_left_hip_roll_03",
                "dof_left_hip_yaw_03",
                "dof_right_hip_roll_03",
                "dof_right_hip_yaw_03",
            ],
            physics_model=physics_model,
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


@attrs.define(frozen=True, kw_only=True)
class ActionRatePenalty(ksim.Reward):
    """Penalty for large per-timestep changes in the action vector (a_t - a_{t-1})."""

    norm: xax.NormType = attrs.field(default="l2", validator=ksim.utils.validators.norm_validator)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        actions = trajectory.action  # (T, dim)
        # Pad one step at the front so we can take a first difference.
        actions_zp = jnp.pad(actions, ((1, 0), (0, 0)), mode="edge")
        # Mask out steps immediately following termination.
        done = jnp.pad(trajectory.done[..., :-1], ((1, 0),), mode="edge")[..., None]

        action_rate = jnp.where(done, 0.0, actions_zp[..., 1:, :] - actions_zp[..., :-1, :])
        # Mean over action dimensions; choose your favourite norm (l1/l2 etc.).
        penalty = xax.get_norm(action_rate, self.norm).mean(axis=-1)
        return penalty


@attrs.define(frozen=True, kw_only=True)
class FollowVelocityReward(ksim.Reward):
    """Reward for tracking the commanded body-frame velocity/rotation.

    The reward is computed for a single component (x, y, or yaw). First
    convert the robot linear / angular velocity to the body frame and then
    penalise the squared error to the corresponding commanded value using an
    exponential kernel:
        r = exp(-error_scale * (v - v_cmd)^2)
    """

    component: str = attrs.field(validator=attrs.validators.in_({"x", "y", "z"}))
    """Which component to track:  ``'x'`` → forward velocity, ``'y'`` → lateral
    velocity, ``'z'`` (treated as yaw) → yaw angular velocity."""
    error_scale: float = attrs.field(default=5.0)
    command_name: str = attrs.field(default="velocity_command")

    def _linear_velocity_body(self, traj: ksim.Trajectory) -> Array:
        linvel_world = traj.qvel[..., :3]
        linvel_body = xax.rotate_vector_by_quat(linvel_world, traj.qpos[..., 3:7], inverse=True)
        return linvel_body

    def _angular_velocity_body(self, traj: ksim.Trajectory) -> Array:
        angvel_world = traj.qvel[..., 3:6]
        angvel_body = xax.rotate_vector_by_quat(angvel_world, traj.qpos[..., 3:7], inverse=True)
        return angvel_body

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:  # noqa: D401 – simple enough
        cmd = trajectory.command[self.command_name][..., :3]  # shape (..., 3) [cx, cy, cyaw]

        if self.component == "x":
            vel_body = self._linear_velocity_body(trajectory)[..., 0]
            cmd_val = cmd[..., 0]
        elif self.component == "y":
            vel_body = self._linear_velocity_body(trajectory)[..., 1]
            cmd_val = cmd[..., 1]
        else:  # "z" → yaw rate around Z
            vel_body = self._angular_velocity_body(trajectory)[..., 2]
            cmd_val = cmd[..., 2]

        error = vel_body - cmd_val
        return jnp.exp(-self.error_scale * jnp.square(error))

    # Provide a more descriptive metric name for logging.
    def get_name(self) -> str:
        comp_name = {"x": "follow_instructed_x", "y": "follow_instructed_y", "z": "follow_instructed_yaw"}[
            self.component
        ]
        return comp_name


# I cant get tensorboard to log 3 different graphs if I directly call FollowVelocityReward.
@attrs.define(frozen=True, kw_only=True)
class FollowVelocityXReward(FollowVelocityReward):
    """Track commanded forward (x) velocity."""

    component: str = attrs.field(default="x", init=False)


@attrs.define(frozen=True, kw_only=True)
class FollowVelocityYReward(FollowVelocityReward):
    """Track commanded lateral (y) velocity."""

    component: str = attrs.field(default="y", init=False)


@attrs.define(frozen=True, kw_only=True)
class FollowVelocityYawReward(FollowVelocityReward):
    """Track commanded yaw rate."""

    component: str = attrs.field(default="z", init=False)


class Actor(eqx.Module):
    """Actor for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
    output_proj: eqx.nn.Linear
    num_inputs: int = eqx.static_field()
    num_outputs: int = eqx.static_field()
    num_mixtures: int = eqx.static_field()
    min_std: float = eqx.static_field()
    max_std: float = eqx.static_field()
    var_scale: float = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        num_outputs: int,
        min_std: float,
        max_std: float,
        var_scale: float,
        hidden_size: int,
        num_mixtures: int,
        depth: int,
    ) -> None:
        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layer
        key, rnn_key = jax.random.split(key)
        rnn_keys = jax.random.split(rnn_key, depth)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for rnn_key in rnn_keys
            ]
        )

        # Project to output
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=num_outputs * 3 * num_mixtures,
            key=key,
        )

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_mixtures = num_mixtures
        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale

    def forward(self, obs_n: Array, carry: Array) -> tuple[distrax.Distribution, Array]:
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            x_n = rnn(x_n, carry[i])
            out_carries.append(x_n)
        out_n = self.output_proj(x_n)

        # Reshape the output to be a mixture of gaussians.
        slice_len = self.num_outputs * self.num_mixtures
        mean_nm = out_n[..., :slice_len].reshape(self.num_outputs, self.num_mixtures)
        std_nm = out_n[..., slice_len : slice_len * 2].reshape(self.num_outputs, self.num_mixtures)
        logits_nm = out_n[..., slice_len * 2 :].reshape(self.num_outputs, self.num_mixtures)

        # Softplus and clip to ensure positive standard deviations.
        std_nm = jnp.clip((jax.nn.softplus(std_nm) + self.min_std) * self.var_scale, max=self.max_std)

        # Apply bias to the means.
        mean_nm = mean_nm + jnp.array([v for _, v in ZEROS])[:, None]

        dist_n = ksim.MixtureOfGaussians(means_nm=mean_nm, stds_nm=std_nm, logits_nm=logits_nm)

        return dist_n, jnp.stack(out_carries, axis=0)


class Critic(eqx.Module):
    """Critic for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
    output_proj: eqx.nn.Linear
    num_inputs: int = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        hidden_size: int,
        depth: int,
    ) -> None:
        num_outputs = 1

        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layer
        key, rnn_key = jax.random.split(key)
        rnn_keys = jax.random.split(rnn_key, depth)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for rnn_key in rnn_keys
            ]
        )

        # Project to output
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=num_outputs,
            key=key,
        )

        self.num_inputs = num_inputs

    def forward(self, obs_n: Array, carry: Array) -> tuple[Array, Array]:
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            x_n = rnn(x_n, carry[i])
            out_carries.append(x_n)
        out_n = self.output_proj(x_n)

        return out_n, jnp.stack(out_carries, axis=0)


class Model(eqx.Module):
    actor: Actor
    critic: Critic

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_actor_inputs: int,
        num_actor_outputs: int,
        num_critic_inputs: int,
        min_std: float,
        max_std: float,
        var_scale: float,
        hidden_size: int,
        num_mixtures: int,
        depth: int,
    ) -> None:
        actor_key, critic_key = jax.random.split(key)
        self.actor = Actor(
            actor_key,
            num_inputs=num_actor_inputs,
            num_outputs=num_actor_outputs,
            min_std=min_std,
            max_std=max_std,
            var_scale=var_scale,
            hidden_size=hidden_size,
            num_mixtures=num_mixtures,
            depth=depth,
        )
        self.critic = Critic(
            critic_key,
            hidden_size=hidden_size,
            depth=depth,
            num_inputs=num_critic_inputs,
        )


class HumanoidWalkingTask(ksim.PPOTask[HumanoidWalkingTaskConfig]):
    def get_optimizer(self) -> optax.GradientTransformation:
        return (
            optax.adam(self.config.learning_rate)
            if self.config.adam_weight_decay == 0.0
            else optax.adamw(self.config.learning_rate, weight_decay=self.config.adam_weight_decay)
        )

    def get_mujoco_model(self) -> mujoco.MjModel:
        mjcf_path = asyncio.run(ksim.get_mujoco_model_path("kbot", name="robot"))
        return mujoco_scenes.mjcf.load_mjmodel(mjcf_path, scene="smooth")

    def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> ksim.Metadata:
        metadata = asyncio.run(ksim.get_mujoco_model_metadata("kbot"))
        if metadata.joint_name_to_metadata is None:
            raise ValueError("Joint metadata is not available")
        if metadata.actuator_type_to_metadata is None:
            raise ValueError("Actuator metadata is not available")
        return metadata

    def get_actuators(
        self,
        physics_model: ksim.PhysicsModel,
        metadata: ksim.Metadata | None = None,
    ) -> ksim.Actuators:
        assert metadata is not None, "Metadata is required"
        return ksim.PositionActuators(
            physics_model=physics_model,
            metadata=metadata,
        )

    def get_physics_randomizers(self, physics_model: ksim.PhysicsModel) -> list[ksim.PhysicsRandomizer]:
        return [
            ksim.StaticFrictionRandomizer(),
            ksim.ArmatureRandomizer(),
            ksim.AllBodiesMassMultiplicationRandomizer(scale_lower=0.95, scale_upper=1.05),
            ksim.JointDampingRandomizer(),
            ksim.JointZeroPositionRandomizer(scale_lower=math.radians(-2), scale_upper=math.radians(2)),
        ]

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        return [
            ksim.PushEvent(
                x_linvel=1.0,
                y_linvel=1.0,
                z_linvel=0.3,
                vel_range=(0.5, 1.0),
                x_angvel=0.0,
                y_angvel=0.0,
                z_angvel=0.0,
                interval_range=(0.5, 4.0),
            ),
        ]

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
            ksim.RandomJointPositionReset.create(physics_model, {k: v for k, v in ZEROS}, scale=0.1),
            ksim.RandomJointVelocityReset(),
        ]

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        return [
            ksim.TimestepObservation(),
            ksim.JointPositionObservation(noise=math.radians(2)),
            ksim.JointVelocityObservation(noise=math.radians(10)),
            ksim.ActuatorForceObservation(),
            ksim.CenterOfMassInertiaObservation(),
            ksim.CenterOfMassVelocityObservation(),
            ksim.BasePositionObservation(),
            ksim.BaseOrientationObservation(),
            ksim.BaseLinearVelocityObservation(),
            ksim.BaseAngularVelocityObservation(),
            ksim.BaseLinearAccelerationObservation(),
            ksim.BaseAngularAccelerationObservation(),
            ksim.ActuatorAccelerationObservation(),
            ksim.ProjectedGravityObservation.create(
                physics_model=physics_model,
                framequat_name="imu_site_quat",
                lag_range=(0.0, 0.1),
                noise=math.radians(1),
            ),
            ksim.SensorObservation.create(
                physics_model=physics_model,
                sensor_name="imu_acc",
                noise=1.0,
            ),
            ksim.SensorObservation.create(
                physics_model=physics_model,
                sensor_name="imu_gyro",
                noise=math.radians(10),
            ),
        ]

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        return [VelocityCommand(dt=self.config.dt)]

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            # Standard rewards.
            ksim.NaiveForwardReward(clip_max=1.25, in_robot_frame=False, scale=3.0),
            ksim.NaiveForwardOrientationReward(scale=1.0),
            # ksim.StayAliveReward(scale=1.0),
            ksim.UprightReward(scale=0.5),
            # Avoid movement penalties.
            # ksim.AngularVelocityPenalty(index=("x", "y"), scale=-0.1),
            # ksim.LinearVelocityPenalty(index=("z"), scale=-0.1),
            # Normalization penalties.
            # ksim.AvoidLimitsPenalty.create(physics_model, scale=-0.01),
            # ksim.JointAccelerationPenalty(scale=-0.01, scale_by_curriculum=True),
            # ksim.JointJerkPenalty(scale=-0.01, scale_by_curriculum=True),
            # ksim.LinkAccelerationPenalty(scale=-0.01, scale_by_curriculum=True),
            # ksim.LinkJerkPenalty(scale=-0.01, scale_by_curriculum=True),
            # ksim.ActionAccelerationPenalty(scale=-0.01, scale_by_curriculum=True),
            # Bespoke rewards.
            BentArmPenalty.create_penalty(physics_model, scale=-0.1),
            StraightLegPenalty.create_penalty(physics_model, scale=-0.1),
            # === Command-following rewards ===
            FollowVelocityXReward(scale=0.15),
            FollowVelocityYReward(scale=0.15),
            FollowVelocityYawReward(scale=0.1),
        ]

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [
            ksim.BadZTermination(unhealthy_z_lower=0.6, unhealthy_z_upper=1.2),
            ksim.FarFromOriginTermination(max_dist=10.0),
        ]

    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> ksim.Curriculum:
        return ksim.DistanceFromOriginCurriculum(
            min_level_steps=5,
        )

    def get_model(self, key: PRNGKeyArray) -> Model:
        return Model(
            key,
            num_actor_inputs=54 if self.config.use_acc_gyro else 48,
            num_actor_outputs=len(ZEROS),
            num_critic_inputs=449,
            min_std=0.001,
            max_std=1.0,
            var_scale=self.config.var_scale,
            hidden_size=self.config.hidden_size,
            num_mixtures=self.config.num_mixtures,
            depth=self.config.depth,
        )

    def run_actor(
        self,
        model: Actor,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[distrax.Distribution, Array]:
        time_1 = observations["timestep_observation"]
        joint_pos_n = observations["joint_position_observation"]
        joint_vel_n = observations["joint_velocity_observation"]
        proj_grav_3 = observations["projected_gravity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        vel_cmd_3 = VelocityCommand.to_policy(commands["velocity_command"])

        obs = [
            jnp.sin(time_1),
            jnp.cos(time_1),
            joint_pos_n,  # NUM_JOINTS
            joint_vel_n,  # NUM_JOINTS
            proj_grav_3,  # 3
            vel_cmd_3,  # 3
        ]
        if self.config.use_acc_gyro:
            obs += [
                imu_acc_3,  # 3
                imu_gyro_3,  # 3
            ]

        obs_n = jnp.concatenate(obs, axis=-1)
        action, carry = model.forward(obs_n, carry)

        return action, carry

    def run_critic(
        self,
        model: Critic,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[Array, Array]:
        time_1 = observations["timestep_observation"]
        dh_joint_pos_j = observations["joint_position_observation"]
        dh_joint_vel_j = observations["joint_velocity_observation"]
        com_inertia_n = observations["center_of_mass_inertia_observation"]
        com_vel_n = observations["center_of_mass_velocity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        proj_grav_3 = observations["projected_gravity_observation"]
        act_frc_obs_n = observations["actuator_force_observation"]
        base_pos_3 = observations["base_position_observation"]
        base_quat_4 = observations["base_orientation_observation"]
        vel_cmd_3 = VelocityCommand.to_policy(commands["velocity_command"])

        obs_n = jnp.concatenate(
            [
                jnp.sin(time_1),
                jnp.cos(time_1),
                dh_joint_pos_j,  # NUM_JOINTS
                dh_joint_vel_j / 10.0,  # NUM_JOINTS
                com_inertia_n,  # 160
                com_vel_n,  # 96
                imu_acc_3,  # 3
                imu_gyro_3,  # 3
                proj_grav_3,  # 3
                act_frc_obs_n / 100.0,  # NUM_JOINTS
                base_pos_3,  # 3
                base_quat_4,  # 4
                vel_cmd_3,  # 3
            ],
            axis=-1,
        )

        return model.forward(obs_n, carry)

    def _model_scan_fn(
        self,
        actor_critic_carry: tuple[Array, Array],
        xs: tuple[ksim.Trajectory, PRNGKeyArray],
        model: Model,
    ) -> tuple[tuple[Array, Array], ksim.PPOVariables]:
        transition, rng = xs

        actor_carry, critic_carry = actor_critic_carry
        actor_dist, next_actor_carry = self.run_actor(
            model=model.actor,
            observations=transition.obs,
            commands=transition.command,
            carry=actor_carry,
        )

        # Gets the log probabilities of the action.
        log_probs = actor_dist.log_prob(transition.action)
        assert isinstance(log_probs, Array)

        value, next_critic_carry = self.run_critic(
            model=model.critic,
            observations=transition.obs,
            commands=transition.command,
            carry=critic_carry,
        )

        transition_ppo_variables = ksim.PPOVariables(
            log_probs=log_probs,
            values=value.squeeze(-1),
        )

        next_carry = jax.tree.map(
            lambda x, y: jnp.where(transition.done, x, y),
            self.get_initial_model_carry(rng),
            (next_actor_carry, next_critic_carry),
        )

        return next_carry, transition_ppo_variables

    def get_ppo_variables(
        self,
        model: Model,
        trajectory: ksim.Trajectory,
        model_carry: tuple[Array, Array],
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PPOVariables, tuple[Array, Array]]:
        scan_fn = functools.partial(self._model_scan_fn, model=model)
        next_model_carry, ppo_variables = xax.scan(
            scan_fn,
            model_carry,
            (trajectory, jax.random.split(rng, len(trajectory.done))),
            jit_level=4,
        )
        return ppo_variables, next_model_carry

    def get_initial_model_carry(self, rng: PRNGKeyArray) -> tuple[Array, Array]:
        return (
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
        )

    def sample_action(
        self,
        model: Model,
        model_carry: tuple[Array, Array],
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
        argmax: bool,
    ) -> ksim.Action:
        actor_carry_in, critic_carry_in = model_carry
        action_dist_j, actor_carry = self.run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
            carry=actor_carry_in,
        )
        action_j = action_dist_j.mode() if argmax else action_dist_j.sample(seed=rng)
        return ksim.Action(action=action_j, carry=(actor_carry, critic_carry_in))


if __name__ == "__main__":
    HumanoidWalkingTask.launch(
        HumanoidWalkingTaskConfig(
            # Training parameters.
            num_envs=4,
            batch_size=2,
            num_passes=4,
            epochs_per_log_step=1,
            rollout_length_seconds=8.0,
            global_grad_clip=2.0,
            # Simulation parameters.
            dt=0.002,
            ctrl_dt=0.02,
            iterations=8,
            ls_iterations=8,
            action_latency_range=(0.003, 0.01),  # Simulate 3-10ms of latency.
            drop_action_prob=0.05,  # Drop 5% of commands.
            # Visualization parameters.
            render_markers=True,
            render_track_body_id=0,
            # Checkpointing parameters.
            save_every_n_seconds=60,
        ),
    )
