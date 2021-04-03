from gym.envs.registration import register

register(
    'HalfCheetah_FLVel-v0',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.half_cheetah:HalfCheetahVelEnv_FL'},
    max_episode_steps=1000
)

register(
    'HalfCheetah_FLBias-v0',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.half_cheetah:HalfCheetahEnv_Bias'},
    max_episode_steps=1000
)


register(
    'Hopper_FLBias-v0',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.hopper:HopperEnv_Bias'},
    max_episode_steps=1000
)

register(
    'Walker2d_FLBias-v0',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.walker2d:Walker2dEnv_Bias'},
    max_episode_steps=1000
)

register(
    'Humanoid_FLBias-v0',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.humanoid:HumanoidEnv_Bias'},
    max_episode_steps=1000
)


# 2D Navigation
# ----------------------------------------

register(
    '2DNavigation-v0',
    entry_point='envs.navigation:Navigation2DEnv',
    max_episode_steps=100
)

register(
    '2DNavigation-v1',
    entry_point='envs.navigation:Navigation2DEnv_FL',
    max_episode_steps=100
)
