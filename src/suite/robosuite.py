from collections import deque
from typing import Any, NamedTuple

from gym import spaces

import numpy as np
import dm_env
from dm_env import StepType, specs, TimeStep
import robosuite as suite
from robosuite.controllers import load_controller_config


MAX_PATH_LENGTH = {
    'Door': 80,
    'Lift': 80,
    'TwoArmPegInHole': 80,
    'PickPlaceCan': 150,
    'NutAssemblySquare': 160,
}


class RGBArrayAsObservationWrapper(dm_env.Environment):
    """
    Use env.render(rgb_array) as observation
    rather than the observation environment provides

    From: https://github.com/hill-a/stable-baselines/issues/915
    """

    def __init__(self, env, max_path_length=125, camera_name='agentview'):
        self._env = env
        self.max_path_length = max_path_length
        self.camera_name = camera_name

        dummy_state = self._env.reset()
        dummy_feat = self.get_low_dim_data(dummy_state)
        dummy_obs = self.get_frame(dummy_state)

        self.observation_space = spaces.Box(low=0, high=255, shape=dummy_obs.shape, dtype=dummy_obs.dtype)
        low, high = self._env.action_spec
        self.action_space = spaces.Box(low=low, high=high)

        # Action spec
        wrapped_action_spec = self.action_space
        if not hasattr(wrapped_action_spec, 'minimum'):
            wrapped_action_spec.minimum = -np.ones(wrapped_action_spec.shape)
        if not hasattr(wrapped_action_spec, 'maximum'):
            wrapped_action_spec.maximum = np.ones(wrapped_action_spec.shape)
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               np.float32,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')
        # Observation spec
        self._obs_spec = {}
        self._obs_spec['pixels'] = specs.BoundedArray(shape=self.observation_space.shape,
                                                      dtype=np.uint8,
                                                      minimum=0,
                                                      maximum=255,
                                                      name='observation')
        self._obs_spec['features'] = specs.Array(shape=dummy_feat.shape,
                                                 dtype=np.float32,
                                                 name='observation')

    def get_low_dim_data(self, obs_dict):
        keys = ['object-state']
        for idx in range(len(self._env.robots)):
            keys += ["robot{}_proprio-state".format(idx)]

        ob_lst = []
        for key in keys:
            if key in obs_dict:
                ob_lst.append(np.array(obs_dict[key]).flatten())
        return np.concatenate(ob_lst)

    def get_frame(self, obs_dict):
        frame = obs_dict[self.camera_name + '_image']
        frame = frame[::-1, :, :]
        return frame

    def reset(self, **kwargs):
        self.episode_step = 0
        state = self._env.reset()
        obs = {}
        obs['features'] = self.get_low_dim_data(state).astype(np.float32)
        obs['pixels'] = self.get_frame(state)
        obs['goal_achieved'] = False

        self.curr_frame = obs['pixels']
        return obs

    def step(self, action):
        state, reward, done, info = self._env.step(action)
        obs = {}
        obs['features'] = self.get_low_dim_data(state).astype(np.float32)
        obs['pixels'] = self.get_frame(state)
        obs['goal_achieved'] = self._env._check_success()
        self.episode_step += 1
        if self.episode_step == self.max_path_length:
            done = True

        self.curr_frame = obs['pixels']
        return obs, reward, done, info

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._action_spec

    def render(self, mode="rgb_array", width=256, height=256):
        return self.curr_frame

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)

        wrapped_obs_spec = env.observation_spec()['pixels']

        pixels_shape = wrapped_obs_spec.shape
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = {}
        self._obs_spec['pixels'] = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name='observation')
        self._obs_spec['features'] = env.observation_spec()['features']

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = {}
        obs['features'] = time_step.observation['features']
        obs['pixels'] = np.concatenate(list(self._frames), axis=0)
        obs['goal_achieved'] = time_step.observation['goal_achieved']
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation['pixels']
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        self._discount = 1.0

        # Action spec
        wrapped_action_spec = env.action_space
        if not hasattr(wrapped_action_spec, 'minimum'):
            wrapped_action_spec.minimum = -np.ones(wrapped_action_spec.shape)
        if not hasattr(wrapped_action_spec, 'maximum'):
            wrapped_action_spec.maximum = np.ones(wrapped_action_spec.shape)
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               np.float32,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')
        # Observation spec
        self._obs_spec = env.observation_spec()

    def step(self, action):
        action = action.astype(self._env.action_space.dtype)
        # Make time step for action space
        observation, reward, done, info = self._env.step(action)
        reward = reward + 1
        step_type = StepType.LAST if done else StepType.MID
        return TimeStep(
            step_type=step_type,
            reward=reward,
            discount=self._discount,
            observation=observation
        )

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._action_spec

    def reset(self):
        obs = self._env.reset()
        return TimeStep(
            step_type=StepType.FIRST,
            reward=0,
            discount=self._discount,
            observation=obs
        )

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def _replace(self, time_step, observation=None, action=None, reward=None, discount=None):
        if observation is None:
            observation = time_step.observation
        if action is None:
            action = time_step.action
        if reward is None:
            reward = time_step.reward
        if discount is None:
            discount = time_step.discount
        return ExtendedTimeStep(observation=observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=reward,
                                discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def make(name, frame_stack, action_repeat, seed, controller):

    robot, name = name.split('_')

    if "TwoArm" in name:
        robots = [robot, robot]
    else:
        robots = robot

    camera_name = 'agentview'
    image_size = 224

    env = suite.make(
        env_name=name,
        robots=robots,
        controller_configs=load_controller_config(default_controller=controller),
        env_configuration="single-arm-opposed",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        reward_shaping=True,  # use dense rewards
        control_freq=20,
        camera_names=camera_name,
        camera_heights=image_size,
        camera_widths=image_size,
    )

    env = RGBArrayAsObservationWrapper(env,
                                       max_path_length=MAX_PATH_LENGTH[name],
                                       camera_name=camera_name)
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = FrameStackWrapper(env, frame_stack)
    env = ExtendedTimeStepWrapper(env)
    return env




