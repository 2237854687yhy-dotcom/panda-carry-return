import argparse
import pickle
from pathlib import Path

import jax
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from train_panda_ab import EPISODE_LENGTH
from train_panda_ab import FINAL_POLICY_PATH
from train_panda_ab import HOVER_A
from train_panda_ab import HOVER_B
from train_panda_ab import START_A
from train_panda_ab import make_env
from train_panda_ab import make_ppo_params
from train_panda_ab import make_train_fn


PHASE_NAMES = {
    0: "grasp at A",
    1: "lift",
    2: "carry to B",
    3: "hold at B",
    4: "return to A",
    5: "hold at A",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record a real MuJoCo rollout from a trained Panda policy."
    )
    parser.add_argument("--policy", default=FINAL_POLICY_PATH)
    parser.add_argument("--output", default="assets/demo.gif")
    parser.add_argument("--stage", default="polish")
    parser.add_argument("--steps", type=int, default=EPISODE_LENGTH)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of using the deterministic policy mode.",
    )
    return parser.parse_args()


def load_trained_policy(policy_path, stage, deterministic):
    policy_path = Path(policy_path)
    if not policy_path.exists():
        raise FileNotFoundError(
            f"{policy_path} does not exist. Run training first, or pass "
            "`--policy /path/to/policy.pkl`."
        )

    with policy_path.open("rb") as f:
        params = pickle.load(f)

    env = make_env(stage)
    ppo_params = make_ppo_params(num_timesteps=0)
    train_fn = make_train_fn(ppo_params, stage, restore_params=params)
    make_inference_fn, restored_params, _ = train_fn(environment=env)
    policy = make_inference_fn(restored_params, deterministic=deterministic)
    return env, policy


def make_camera():
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.array([0.42, 0.10, 0.18])
    camera.distance = 1.75
    camera.azimuth = 135
    camera.elevation = -24
    return camera


def font(size, bold=False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        if candidate:
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                pass
    return ImageFont.load_default()


def scalar(value, default=0.0):
    if value is None:
        return default
    return float(np.asarray(value))


def annotate(frame, state, step):
    image = Image.fromarray(frame)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    phase = int(np.asarray(state.info["task_phase"]))
    hold_steps = int(np.asarray(state.info["hold_steps"]))
    reward = scalar(state.reward)
    success = scalar(state.metrics.get("task_success"))
    grasped = scalar(state.metrics.get("grasped"))
    at_b = scalar(state.metrics.get("at_b"))
    at_a = scalar(state.metrics.get("at_a"))

    title_font = font(25, bold=True)
    text_font = font(18)
    small_font = font(15)

    draw.rounded_rectangle((24, 22, 355, 128), radius=10, fill=(15, 23, 42, 210))
    draw.text((44, 38), "Panda Carry Return", font=title_font, fill=(255, 255, 255))
    draw.text(
        (44, 74),
        f"{step:03d} | {PHASE_NAMES.get(phase, 'done')} | hold {hold_steps}",
        font=text_font,
        fill=(226, 232, 240),
    )
    draw.text(
        (44, 101),
        (
            f"reward {reward:.2f}  grasp {grasped:.0f}  "
            f"B {at_b:.0f}  A {at_a:.0f}  success {success:.0f}"
        ),
        font=small_font,
        fill=(203, 213, 225),
    )

    marker_y = image.height - 62
    markers = [
        ("A start", START_A, (239, 68, 68, 235)),
        ("A hover", HOVER_A, (248, 113, 113, 235)),
        ("B hover", HOVER_B, (37, 99, 235, 235)),
    ]
    x = 28
    for label, _, color in markers:
        draw.ellipse((x, marker_y, x + 18, marker_y + 18), fill=color)
        draw.text((x + 27, marker_y - 1), label, font=small_font, fill=(15, 23, 42))
        x += 120

    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def record_rollout(env, policy, args):
    rng = jax.random.PRNGKey(args.seed)
    state = env.reset(rng)

    jit_policy = jax.jit(policy)
    jit_step = jax.jit(env.step)

    model = env.mj_model
    data = mujoco.MjData(model)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = make_camera()

    frames = []
    for step in range(args.steps):
        rng, act_rng = jax.random.split(rng)
        action, _ = jit_policy(state.obs, act_rng)
        state = jit_step(state, action)
        state.reward.block_until_ready()

        if step % args.stride == 0:
            data.qpos[:] = np.asarray(state.data.qpos)
            data.qvel[:] = np.asarray(state.data.qvel)
            data.mocap_pos[:] = np.asarray(state.data.mocap_pos)
            data.mocap_quat[:] = np.asarray(state.data.mocap_quat)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frames.append(annotate(renderer.render(), state, step))

        if bool(np.asarray(state.done)):
            break

    renderer.close()
    return frames


def save_gif(frames, output, fps):
    if not frames:
        raise RuntimeError("No frames were recorded.")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(1000 / fps))
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    return output


def main():
    args = parse_args()
    env, policy = load_trained_policy(
        args.policy, args.stage, deterministic=not args.stochastic
    )
    frames = record_rollout(env, policy, args)
    output = save_gif(frames, args.output, args.fps)
    print(f"Saved {len(frames)} frames to {output}")


if __name__ == "__main__":
    main()
