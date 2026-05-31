from datetime import datetime
import functools
import pickle
import time

import jax
import jax.numpy as jp
import numpy as np
import mujoco
import mujoco.viewer
from mujoco import mjx

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import manipulation_params


ENV_NAME = "PandaPickCube"

EPISODE_LENGTH = 150
HOLD_STEPS = 12

START_A = jp.array([0.45, 0.00, 0.03])
HOVER_A = jp.array([0.45, 0.00, 0.16])
HOVER_B = jp.array([0.45, 0.25, 0.16])
TARGET_QUAT = jp.array([1.0, 0.0, 0.0, 0.0])

PHASE_TARGETS = jp.stack([
    START_A,   # 0: reach and grasp the cube at A
    HOVER_A,   # 1: lift while keeping the cube grasped
    HOVER_B,   # 2: carry to B
    HOVER_B,   # 3: hold at B
    HOVER_A,   # 4: return to A
    HOVER_A,   # 5: hold at A
])

STAGE_MAX_PHASE = {
    "grasp": 0,
    "lift": 1,
    "carry_b": 2,
    "hold_b": 3,
    "return_a": 5,
    "polish": 5,
}

TRAINING_STAGES = (
    ("grasp", 500_000),
    ("lift", 1_000_000),
    ("carry_b", 1_500_000),
    ("hold_b", 1_000_000),
    ("return_a", 2_000_000),
    ("polish", 1_000_000),
)

FINAL_POLICY_PATH = "panda_policy_grasp_b_return_a.pkl"


class PandaCarryReturnEnv:
    """Grasp at A, hold at B, then carry back and hold at A."""

    def __init__(self, base_env, stage):
        if stage not in STAGE_MAX_PHASE:
            raise ValueError(f"Unknown stage: {stage}")

        self.base_env = base_env
        self.stage = stage
        self.max_phase = STAGE_MAX_PHASE[stage]
        self.base_reset_rng = jax.random.PRNGKey(0)

        if hasattr(self.base_env, "_obj_qposadr"):
            self.box_qpos_adr = int(self.base_env._obj_qposadr)
        else:
            box_body_id = mujoco.mj_name2id(
                self.base_env.mj_model,
                mujoco.mjtObj.mjOBJ_BODY,
                "box",
            )
            if box_body_id < 0:
                raise ValueError("找不到名为 'box' 的 body。")
            if self.base_env.mj_model.body_jntnum[box_body_id] == 0:
                raise ValueError("名为 'box' 的 body 没有关节，无法设置 qpos。")
            box_joint_id = self.base_env.mj_model.body_jntadr[box_body_id]
            self.box_qpos_adr = int(
                self.base_env.mj_model.jnt_qposadr[box_joint_id]
            )

        box_joint_id = self.base_env.mj_model.body("box").jntadr[0]
        self.box_qvel_adr = int(self.base_env.mj_model.jnt_dofadr[box_joint_id])
        self.finger_qposadr = self.base_env._robot_qposadr[-2:]
        self.arm_qposadr = self.base_env._robot_arm_qposadr
        self.mocap_target = self.base_env._mocap_target

    def __getattr__(self, name):
        return getattr(self.base_env, name)

    def _phase_target(self, phase):
        phase = jp.clip(phase, 0, PHASE_TARGETS.shape[0] - 1)
        return PHASE_TARGETS[phase.astype(jp.int32)]

    def _with_target(self, data, target_pos):
        return data.replace(
            mocap_pos=data.mocap_pos.at[self.mocap_target, :].set(target_pos),
            mocap_quat=data.mocap_quat.at[self.mocap_target, :].set(TARGET_QUAT),
        )

    def reset(self, rng):
        state = self.base_env.reset(self.base_reset_rng)

        qpos = state.data.qpos
        qpos = qpos.at[self.box_qpos_adr + 0].set(START_A[0])
        qpos = qpos.at[self.box_qpos_adr + 1].set(START_A[1])
        qpos = qpos.at[self.box_qpos_adr + 2].set(START_A[2])
        qpos = qpos.at[self.box_qpos_adr + 3].set(TARGET_QUAT[0])
        qpos = qpos.at[self.box_qpos_adr + 4].set(TARGET_QUAT[1])
        qpos = qpos.at[self.box_qpos_adr + 5].set(TARGET_QUAT[2])
        qpos = qpos.at[self.box_qpos_adr + 6].set(TARGET_QUAT[3])

        phase = jp.array(0, dtype=jp.int32)
        hold_steps = jp.array(0, dtype=jp.int32)
        target_pos = self._phase_target(phase)

        info = dict(state.info)
        info["target_a"] = HOVER_A
        info["target_b"] = HOVER_B
        info["target_pos"] = target_pos
        info["task_phase"] = phase
        info["hold_steps"] = hold_steps
        info["reached_box"] = jp.array(0.0)

        data = state.data.replace(qpos=qpos)
        data = self._with_target(data, target_pos)
        data = mjx.forward(self.base_env.mjx_model, data)
        obs = self.base_env._get_obs(data, info)
        metrics = self._initial_metrics(state.metrics)

        return state.replace(data=data, obs=obs, metrics=metrics, info=info)

    def step(self, state, action):
        phase = state.info["task_phase"]
        hold_steps = state.info["hold_steps"]

        if "steps" in state.info:
            is_new_episode = state.info["steps"] == 0
            phase = jp.where(is_new_episode, jp.zeros_like(phase), phase)
            hold_steps = jp.where(
                is_new_episode, jp.zeros_like(hold_steps), hold_steps
            )

        target_pos = self._phase_target(phase)
        info = dict(state.info)
        info["task_phase"] = phase
        info["hold_steps"] = hold_steps
        info["target_pos"] = target_pos

        data = self._with_target(state.data, target_pos)
        state = state.replace(data=data, info=info)

        next_state = self.base_env.step(state, action)
        no_floor_collision = next_state.metrics["no_floor_collision"]
        terms = self._terms(next_state.data, action, no_floor_collision)

        stable_target = jp.where(
            phase == 3,
            terms["stable_at_b"],
            jp.where(phase == 5, terms["stable_at_a"], jp.array(0.0)),
        )
        hold_steps_candidate = jp.where(
            stable_target > 0.5,
            hold_steps + 1,
            jp.zeros_like(hold_steps),
        )
        terms["hold_b_done"] = (
            (phase == 3) & (hold_steps_candidate >= HOLD_STEPS)
        ).astype(float)
        terms["hold_a_done"] = (
            (phase == 5) & (hold_steps_candidate >= HOLD_STEPS)
        ).astype(float)
        terms["task_success"] = terms["hold_a_done"]

        next_phase = self._advance_phase(phase, terms)
        next_phase = jp.minimum(next_phase, self.max_phase)
        next_hold_steps = jp.where(
            next_phase == phase,
            hold_steps_candidate,
            jp.zeros_like(hold_steps_candidate),
        )
        terms["hold_steps"] = next_hold_steps.astype(float)

        next_target_pos = self._phase_target(next_phase)
        next_info = dict(next_state.info)
        next_info["target_a"] = HOVER_A
        next_info["target_b"] = HOVER_B
        next_info["target_pos"] = next_target_pos
        next_info["task_phase"] = next_phase
        next_info["hold_steps"] = next_hold_steps

        next_data = self._with_target(next_state.data, next_target_pos)
        next_obs = self.base_env._get_obs(next_data, next_info)
        reward = self._structured_reward(phase, terms)

        metrics = dict(next_state.metrics)
        metrics.update(terms)
        metrics["task_phase"] = next_phase.astype(float)
        metrics["stage_reward"] = reward

        return next_state.replace(
            data=next_data,
            obs=next_obs,
            reward=reward,
            metrics=metrics,
            info=next_info,
        )

    def _terms(self, data, action, no_floor_collision):
        box_pos = data.xpos[self.base_env._obj_body]
        gripper_pos = data.site_xpos[self.base_env._gripper_site]
        box_linvel = data.qvel[self.box_qvel_adr : self.box_qvel_adr + 3]
        box_angvel = data.qvel[self.box_qvel_adr + 3 : self.box_qvel_adr + 6]

        gripper_dist = jp.linalg.norm(box_pos - gripper_pos)
        finger_gap = jp.sum(data.qpos[self.finger_qposadr])
        box_speed = jp.linalg.norm(box_linvel)
        box_angspeed = jp.linalg.norm(box_angvel)

        gripper_box = 1.0 - jp.tanh(8.0 * gripper_dist)
        finger_closed = 1.0 - jp.tanh(25.0 * jp.maximum(finger_gap - 0.04, 0.0))
        grasp_score = gripper_box * finger_closed
        grasped = (gripper_dist < 0.025) & (finger_gap < 0.075)

        a_dist = jp.linalg.norm(box_pos - HOVER_A)
        b_dist = jp.linalg.norm(box_pos - HOVER_B)
        lift_dist = jp.linalg.norm(box_pos - HOVER_A)
        pick_dist = jp.linalg.norm(box_pos - START_A)

        pick_score = 1.0 - jp.tanh(8.0 * pick_dist)
        lift_score = 1.0 - jp.tanh(7.0 * lift_dist)
        b_score = 1.0 - jp.tanh(7.0 * b_dist)
        a_score = 1.0 - jp.tanh(7.0 * a_dist)
        settle_score = 1.0 - jp.tanh(8.0 * box_speed + 0.5 * box_angspeed)

        lifted = (box_pos[2] > 0.11) & grasped
        at_b = (b_dist < 0.045) & grasped
        at_a = (a_dist < 0.045) & grasped
        stable_at_b = at_b & (box_speed < 0.08) & (box_angspeed < 1.5)
        stable_at_a = at_a & (box_speed < 0.08) & (box_angspeed < 1.5)

        drop_penalty = ((box_pos[2] > 0.08) & (~grasped)).astype(float)
        drop_penalty += jp.maximum(-box_linvel[2] - 0.2, 0.0) * (1.0 - grasp_score)

        posture = 1.0 - jp.tanh(
            jp.linalg.norm(
                data.qpos[self.arm_qposadr]
                - self.base_env._init_q[self.arm_qposadr]
            )
        )
        action_cost = jp.mean(jp.square(action))

        return {
            "gripper_box": gripper_box,
            "finger_closed": finger_closed,
            "grasp_score": grasp_score,
            "pick_score": pick_score,
            "lift_score": lift_score,
            "b_score": b_score,
            "a_score": a_score,
            "settle_score": settle_score,
            "grasped": grasped.astype(float),
            "lifted": lifted.astype(float),
            "at_b": at_b.astype(float),
            "at_a": at_a.astype(float),
            "stable_at_b": stable_at_b.astype(float),
            "stable_at_a": stable_at_a.astype(float),
            "hold_b_done": jp.array(0.0),
            "hold_a_done": jp.array(0.0),
            "task_success": jp.array(0.0),
            "hold_steps": jp.array(0.0),
            "box_speed": box_speed,
            "box_angspeed": box_angspeed,
            "drop_penalty": drop_penalty,
            "posture": posture,
            "action_cost": action_cost,
            "no_floor_collision": no_floor_collision,
        }

    def _initial_metrics(self, base_metrics):
        metrics = dict(base_metrics)
        zero = jp.array(0.0)
        metrics.update({
            "gripper_box": zero,
            "finger_closed": zero,
            "grasp_score": zero,
            "pick_score": zero,
            "lift_score": zero,
            "b_score": zero,
            "a_score": zero,
            "settle_score": zero,
            "grasped": zero,
            "lifted": zero,
            "at_b": zero,
            "at_a": zero,
            "stable_at_b": zero,
            "stable_at_a": zero,
            "hold_b_done": zero,
            "hold_a_done": zero,
            "task_success": zero,
            "hold_steps": zero,
            "box_speed": zero,
            "box_angspeed": zero,
            "drop_penalty": zero,
            "posture": zero,
            "action_cost": zero,
            "task_phase": zero,
            "stage_reward": zero,
        })
        return metrics

    def _advance_phase(self, phase, terms):
        next_phase = phase
        next_phase = jp.where(
            (phase == 0) & (terms["grasped"] > 0.5), 1, next_phase
        )
        next_phase = jp.where(
            (phase == 1) & (terms["lifted"] > 0.5), 2, next_phase
        )
        next_phase = jp.where(
            (phase == 2) & (terms["at_b"] > 0.5), 3, next_phase
        )
        next_phase = jp.where(
            (phase == 3) & (terms["hold_b_done"] > 0.5), 4, next_phase
        )
        next_phase = jp.where(
            (phase == 4) & (terms["at_a"] > 0.5), 5, next_phase
        )
        return next_phase

    def _structured_reward(self, phase, terms):
        grasp_reward = (
            5.0 * terms["gripper_box"]
            + 4.0 * terms["grasp_score"]
            + 2.0 * terms["grasped"]
        )
        lift_reward = (
            2.0 * terms["grasp_score"]
            + 8.0 * terms["lift_score"]
            + 3.0 * terms["lifted"]
            - 3.0 * terms["drop_penalty"]
        )
        carry_b_reward = (
            2.0 * terms["grasp_score"]
            + 10.0 * terms["b_score"]
            + 3.0 * terms["at_b"]
            - 5.0 * terms["drop_penalty"]
        )
        hold_b_reward = (
            2.0 * terms["grasp_score"]
            + 9.0 * terms["b_score"]
            + 3.0 * terms["settle_score"]
            + 6.0 * terms["stable_at_b"]
            + 12.0 * terms["hold_b_done"]
            - 5.0 * terms["drop_penalty"]
        )
        return_a_reward = (
            2.0 * terms["grasp_score"]
            + 10.0 * terms["a_score"]
            + 3.0 * terms["at_a"]
            - 5.0 * terms["drop_penalty"]
        )
        hold_a_reward = (
            2.0 * terms["grasp_score"]
            + 9.0 * terms["a_score"]
            + 3.0 * terms["settle_score"]
            + 6.0 * terms["stable_at_a"]
            + 20.0 * terms["task_success"]
            - 5.0 * terms["drop_penalty"]
        )

        phase_rewards = jp.stack([
            grasp_reward,
            lift_reward,
            carry_b_reward,
            hold_b_reward,
            return_a_reward,
            hold_a_reward,
        ])
        reward = phase_rewards[jp.clip(phase, 0, 5).astype(jp.int32)]
        reward = reward + 0.25 * terms["no_floor_collision"]

        if self.stage == "polish":
            reward = reward + 0.15 * terms["posture"]
            reward = reward - 0.02 * terms["action_cost"]
        else:
            reward = reward - 0.005 * terms["action_cost"]

        return jp.clip(reward, -1e4, 1e4)


def draw_marker(viewer, pos, rgba, radius=0.035):
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        return

    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[viewer.user_scn.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius]),
        np.array(pos),
        np.eye(3).flatten(),
        np.array(rgba),
    )
    viewer.user_scn.ngeom += 1


def make_env(stage):
    env_cfg = registry.get_default_config(ENV_NAME)
    env_cfg.impl = "jax"
    base_env = registry.load(ENV_NAME, config=env_cfg)
    return PandaCarryReturnEnv(base_env, stage)


def make_ppo_params(num_timesteps):
    ppo_params = manipulation_params.brax_ppo_config(ENV_NAME)

    ppo_params.episode_length = EPISODE_LENGTH
    ppo_params.num_timesteps = num_timesteps
    ppo_params.num_envs = 256
    ppo_params.num_eval_envs = 64
    ppo_params.num_evals = 4
    ppo_params.batch_size = 256
    ppo_params.num_minibatches = 16
    ppo_params.num_updates_per_batch = 8
    ppo_params.discounting = 0.97
    ppo_params.learning_rate = 8e-4
    ppo_params.entropy_cost = 1.5e-2

    return ppo_params


def make_progress(stage, start_time):
    def progress(num_steps, metrics):
        reward = metrics["eval/episode_reward"]
        reward_std = metrics["eval/episode_reward_std"]
        phase_sum = metrics.get("eval/episode_task_phase", 0.0)
        phase_avg = phase_sum / EPISODE_LENGTH
        grasped = metrics.get("eval/episode_grasped", 0.0)
        lifted = metrics.get("eval/episode_lifted", 0.0)
        at_b = metrics.get("eval/episode_at_b", 0.0)
        hold_b = metrics.get("eval/episode_stable_at_b", 0.0)
        at_a = metrics.get("eval/episode_at_a", 0.0)
        hold_a = metrics.get("eval/episode_stable_at_a", 0.0)
        success = metrics.get("eval/episode_task_success", 0.0)
        drop_penalty = metrics.get("eval/episode_drop_penalty", 0.0)

        print(
            f"[{stage}] steps={num_steps} | reward={reward:.3f} ± {reward_std:.3f} "
            f"| phase_avg={phase_avg:.2f} | grasp_steps={grasped:.1f} "
            f"| lift_steps={lifted:.1f} | at_b={at_b:.1f} | hold_b={hold_b:.1f} "
            f"| at_a={at_a:.1f} | hold_a={hold_a:.1f} | success={success:.1f} "
            f"| drop={drop_penalty:.2f} | elapsed={datetime.now() - start_time}",
            flush=True,
        )

    return progress


def make_train_fn(ppo_params, stage, restore_params):
    ppo_training_params = dict(ppo_params)
    network_factory = ppo_networks.make_ppo_networks

    if "network_factory" in ppo_params:
        del ppo_training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **ppo_params.network_factory,
        )

    return functools.partial(
        ppo.train,
        **ppo_training_params,
        network_factory=network_factory,
        progress_fn=make_progress(stage, datetime.now()),
        wrap_env_fn=wrapper.wrap_for_brax_training,
        seed=1,
        restore_params=restore_params,
        restore_value_fn=False,
    )


def train_stage(stage, num_timesteps, restore_params):
    print(f"\n=== Start stage: {stage} ({num_timesteps:,} steps) ===")

    env = make_env(stage)
    ppo_params = make_ppo_params(num_timesteps)
    train_fn = make_train_fn(ppo_params, stage, restore_params)

    make_inference_fn, params, metrics = train_fn(environment=env)

    path = f"panda_policy_grasp_b_return_a_{stage}.pkl"
    with open(path, "wb") as f:
        pickle.dump(params, f)

    print(f"Stage saved as {path}")
    print(metrics)
    return make_inference_fn, params, metrics


def train_all_stages():
    params = None
    make_inference_fn = None
    metrics = None

    for stage, num_timesteps in TRAINING_STAGES:
        make_inference_fn, params, metrics = train_stage(
            stage, num_timesteps, params
        )

    with open(FINAL_POLICY_PATH, "wb") as f:
        pickle.dump(params, f)

    print(f"\nFinal policy saved as {FINAL_POLICY_PATH}")
    return make_inference_fn, params, metrics


def launch_viewer(make_inference_fn, params):
    print("Launching MuJoCo viewer...")

    env = make_env("polish")
    inference_fn = make_inference_fn(params)
    jit_inference_fn = jax.jit(inference_fn)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(0)
    fixed_reset_rng = jax.random.PRNGKey(0)
    state = env.reset(fixed_reset_rng)

    model = env.mj_model
    data = mujoco.MjData(model)

    step_count = 0
    max_steps = EPISODE_LENGTH

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            rng, act_rng = jax.random.split(rng)

            action, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, action)
            state.reward.block_until_ready()

            data.qpos[:] = np.array(state.data.qpos)
            data.qvel[:] = np.array(state.data.qvel)
            mujoco.mj_forward(model, data)

            viewer.user_scn.ngeom = 0
            draw_marker(viewer, START_A, [1.0, 0.0, 0.0, 0.8], radius=0.025)
            draw_marker(viewer, HOVER_A, [1.0, 0.0, 0.0, 0.8])
            draw_marker(viewer, HOVER_B, [0.0, 0.2, 1.0, 0.8])
            draw_marker(
                viewer,
                np.array(state.info["target_pos"]),
                [1.0, 0.9, 0.0, 0.8],
                radius=0.025,
            )

            viewer.sync()

            step_count += 1
            if step_count >= max_steps or bool(state.done):
                state = env.reset(fixed_reset_rng)
                step_count = 0

            time.sleep(0.02)


if __name__ == "__main__":
    make_inference_fn, params, metrics = train_all_stages()
    print("Training done.")
    print(metrics)
    launch_viewer(make_inference_fn, params)
