from gymnasium.envs.registration import register

register(
    id='carla-v0',
    entry_point='gym_carla.envs:CarlaAvoidanceEnv',
    kwargs={},
    max_episode_steps=100,
)