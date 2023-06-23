from collections import deque
from typing import Any, NamedTuple

import gym
from gym import spaces

import numpy as np
import dm_env
from dm_env import StepType, specs, TimeStep


CAMERA = {
    'kitchen_knob1_on-v3': 'left_cap2',
    'kitchen_light_on-v3': 'right_cap2',
    'kitchen_sdoor_open-v3': 'right_cap2',
    'kitchen_ldoor_open-v3': 'left_cap2',
    'kitchen_micro_open-v3': 'left_cap2',
}

MAX_PATH_LENGTH = {
    'kitchen_knob1_on-v3': 50,
    'kitchen_light_on-v3': 50,
    'kitchen_sdoor_open-v3': 50,
    'kitchen_ldoor_open-v3': 50,
    'kitchen_micro_open-v3': 50,
}


class RGBArrayAsObservationWrapper(dm_env.Environment):
    """
    Use env.render(rgb_array) as observation
    rather than the observation environment provides

    From: https://github.com/hill-a/stable-baselines/issues/915
    """

    def __init__(self, env, width=84, height=84, max_path_length=125, camera_name='left_cap2'):
        self._env = env
        self._width = width
        self._height = height
        self.max_path_length = max_path_length
        self.camera_name = camera_name

        dummy_feat = self._env.reset()
        dummy_obs = self.get_frame()

        self.observation_space = spaces.Box(low=0, high=255, shape=dummy_obs.shape, dtype=dummy_obs.dtype)
        self.action_space = self._env.action_space

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

    def reset(self, **kwargs):
        #TODO how to process seed?

        self.episode_step = 0

        obs = {}
        obs['features'] = self._env.reset(**kwargs).astype(np.float32)
        obs['pixels'] = self.get_frame()
        obs['goal_achieved'] = False
        return obs

    def step(self, action):
        observation, reward, done, info = self._env.step(action)
        obs = {}
        obs['features'] = observation.astype(np.float32)
        obs['pixels'] = self.get_frame()
        obs['goal_achieved'] = info['solved']
        self.episode_step += 1
        if self.episode_step == self.max_path_length:
            done = True
        return obs, reward, done, info

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._action_spec

    def render(self, mode="rgb_array", width=256, height=256):
        assert mode == 'rgb_array'
        frame = self._env.sim.render(width=self._width, height=self._height, depth=False,
                                     camera_name=self.camera_name)
        frame = frame[::-1, :, :]
        return frame

    def get_frame(self):
        frame = self._env.sim.render(width=self._width, height=self._height, depth=False,
                                     camera_name=self.camera_name)
        frame = frame[::-1, :, :]
        return frame

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


def make(name, frame_stack, action_repeat, seed):

    env = gym.make(name)
    env.seed(seed)

    img_size = 224
    # add wrappers
    env = RGBArrayAsObservationWrapper(env,
                                       width=img_size, height=img_size,
                                       max_path_length=MAX_PATH_LENGTH[name],
                                       camera_name=CAMERA[name])

    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = FrameStackWrapper(env, frame_stack)
    env = ExtendedTimeStepWrapper(env)
    return env










