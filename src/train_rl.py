#!/usr/bin/env python3

import warnings
import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

warnings.filterwarnings('ignore', category=DeprecationWarning)
torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec[cfg.obs_type].shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self.store_repr = self.cfg.store_repr

    def setup(self):
        # create logger
        group_name = self.cfg.exp_prefix + '_' + self.cfg.suite.name + '_' + self.cfg.task_name + '_' + self.cfg.exp_suffix
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb, use_wandb=self.cfg.use_wandb, group_name=group_name)
        # create envs
        self.train_env = hydra.utils.call(self.cfg.suite.task_make_fn)
        self.eval_env = hydra.utils.call(self.cfg.suite.task_make_fn)

        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.cfg.agent)

        # create replay buffer
        data_specs = [
            specs.Array((self.agent.repr_dim,), np.float32, 'observation') if self.cfg.store_repr
            else self.train_env.observation_spec()[self.cfg.obs_type],
            self.train_env.action_spec(),
            specs.Array((1,), np.float32, 'reward'),
            specs.Array((1,), np.float32, 'discount')
        ]

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.suite.save_buffer, self.cfg.nstep, self.cfg.suite.discount)

        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)

        self.cfg.suite.num_train_frames = self.cfg.num_train_frames_drq

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.suite.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self, last_eval=False):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.suite.num_eval_episodes)

        paths = []
        while eval_until_episode(episode):
            path = []
            time_step = self.eval_env.reset()
            if self.store_repr:
                representation = self.agent.get_repr(time_step.observation[self.cfg.obs_type])
                curr_obs = time_step.observation
                curr_obs['pixels'] = representation
                time_step._replace(observation=curr_obs)

            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation[self.cfg.obs_type],
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                if self.store_repr:
                    representation = self.agent.get_repr(time_step.observation[self.cfg.obs_type])
                    curr_obs = time_step.observation
                    curr_obs['pixels'] = representation
                    time_step._replace(observation=curr_obs)

                path.append(time_step.observation['goal_achieved'])
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')
            if self.cfg.suite.name == 'metaworld':
                paths.append(1 if np.sum(path) > 10 else 0)
            elif self.cfg.suite.name == 'robosuite' or self.cfg.suite.name == 'kitchen':
                paths.append(1 if np.sum(path) > 5 else 0)

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.suite.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
            log("success_percentage", np.mean(paths))

        if last_eval:
            episode_reward = total_reward / episode
            success_percentage = np.mean(paths)
            return episode_reward, success_percentage

    def train_rl(self):
        # predicates
        train_until_step = utils.Until(self.cfg.suite.num_train_frames,
                                       self.cfg.suite.action_repeat)
        seed_until_step = utils.Until(self.cfg.suite.num_seed_frames,
                                      self.cfg.suite.action_repeat)
        eval_every_step = utils.Every(self.cfg.suite.eval_every_frames,
                                      self.cfg.suite.action_repeat)

        episode_step, episode_reward = 0, 0

        time_steps = list()
        observations = list()
        actions = list()

        time_step = self.train_env.reset()
        if self.store_repr:
            representation = self.agent.get_repr(time_step.observation[self.cfg.obs_type])
            curr_obs = time_step.observation
            curr_obs['pixels'] = representation
            time_step._replace(observation=curr_obs)
        time_steps.append(time_step)
        observations.append(time_step.observation[self.cfg.obs_type])
        actions.append(time_step.action)

        self.train_video_recorder.init(time_step.observation[self.cfg.obs_type])
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                if self._global_episode % 10 == 0:
                    self.train_video_recorder.save(f'{self.global_frame}.mp4')

                for i, elt in enumerate(time_steps):
                    elt = elt._replace(
                        observation=time_steps[i].observation[self.cfg.obs_type])
                    self.replay_storage.add(elt)

                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.suite.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame, ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # reset env
                time_steps = list()
                observations = list()
                actions = list()

                time_step = self.train_env.reset()
                if self.store_repr:
                    representation = self.agent.get_repr(time_step.observation[self.cfg.obs_type])
                    curr_obs = time_step.observation
                    curr_obs['pixels'] = representation
                    time_step._replace(observation=curr_obs)
                time_steps.append(time_step)
                observations.append(time_step.observation[self.cfg.obs_type])
                actions.append(time_step.action)
                self.train_video_recorder.init(time_step.observation[self.cfg.obs_type])
                # try to save snapshot
                if self.cfg.suite.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation[self.cfg.obs_type],
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                # Update
                metrics = self.agent.update(self.replay_iter, self.global_step)
                if self.global_step % self.cfg.log_freq == 0:
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            if self.store_repr:
                representation = self.agent.get_repr(time_step.observation[self.cfg.obs_type])
                curr_obs = time_step.observation
                curr_obs['pixels'] = representation
                time_step._replace(observation=curr_obs)
            episode_reward += time_step.reward

            time_steps.append(time_step)
            observations.append(time_step.observation[self.cfg.obs_type])
            actions.append(time_step.action)

            self.train_video_recorder.record(time_step.observation[self.cfg.obs_type])
            episode_step += 1
            self._global_step += 1

        # Final eval
        self.logger.log('eval_total_time', self.timer.total_time(), self.global_frame)
        self.cfg.suite.num_eval_episodes = 100
        episode_reward, success_percentage = self.eval(last_eval=True)
        return episode_reward, success_percentage

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        payload.update(self.agent.save_snapshot())
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self, snapshot):
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        agent_payload = {}
        for k, v in payload.items():
            if k not in self.__dict__:
                agent_payload[k] = v
        self.agent.load_snapshot(agent_payload)


@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    from train_rl import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)

    episode_reward, success_percentage = workspace.train_rl()

    utils.write_to_csv(root_dir, cfg, episode_reward, success_percentage)


if __name__ == '__main__':
    main()