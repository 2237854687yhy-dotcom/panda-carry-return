import time
import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer
import numpy as np

from mujoco_playground import registry

ENV_NAME = "PandaPickCubeOrientation"

env_cfg = registry.get_default_config(ENV_NAME)
env_cfg.impl = "jax"

env = registry.load(ENV_NAME, config=env_cfg)

rng = jax.random.PRNGKey(0)
state = env.reset(rng)

step_fn = jax.jit(env.step)

# 预编译一次，第一次会慢，后面会顺
zero_action = jp.zeros((env.action_size,))
state = step_fn(state, zero_action)
state.reward.block_until_ready()

model = env.mj_model
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Viewer started. Close window to stop.")

    while viewer.is_running():
        rng, action_rng = jax.random.split(rng)

        # 小幅随机动作，避免机械臂乱抽
        action = jax.random.uniform(
            action_rng,
            shape=(env.action_size,),
            minval=-0.03,
            maxval=0.03,
        )

        state = step_fn(state, action)
        state.reward.block_until_ready()

        data.qpos[:] = np.array(state.data.qpos)
        data.qvel[:] = np.array(state.data.qvel)

        mujoco.mj_forward(model, data)
        viewer.sync()

        time.sleep(0.04)