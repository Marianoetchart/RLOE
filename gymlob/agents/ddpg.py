import time
import typing

import gym
import numpy as np
import omegaconf
import torch
import wandb

from gymlob.agents.agent import Agent
from gymlob.learners.ddpg import DDPGLearner

from gymlob.utils.replay_buffer import ReplayBuffer
from gymlob.utils.noise import OUNoise
from gymlob.utils.utils import numpy2floattensor


class DDPGAgent(Agent):

    def __init__(self,
                 env: gym.Env,
                 cfg: omegaconf.dictconfig.DictConfig
                 ):

        Agent.__init__(self, env, cfg)

        self.learner = DDPGLearner(self.env, cfg, self.observation_space, self.action_space)

        if not self.cfg.test:
            self.memory = ReplayBuffer(max_len=self.hyper_params.buffer_size,
                                       batch_size=self.hyper_params.batch_size,
                                       gamma=self.hyper_params.gamma)

        self.noise = OUNoise(size=self.action_space.shape[0],
                             theta=self.learner_cfg.noise_cfg.ou_noise_theta,
                             sigma=self.learner_cfg.noise_cfg.ou_noise_sigma)

        self.current_state = np.zeros((1,))

        self.episode_step = 0
        self.i_episode = 0

    def select_action(self,
                      state: np.ndarray
                      ) -> np.ndarray:

        self.current_state = state

        # if initial random action should be conducted
        if self.total_step < self.hyper_params.initial_random_action and not self.cfg.test:
            return np.array(self.action_space.sample())
        else:
            with torch.no_grad():
                state = numpy2floattensor(state, self.learner.device)
                selected_action = self.learner.actor(state).detach().cpu().numpy()

            if not self.cfg.test:
                noise = self.noise.sample()
                selected_action = np.clip(selected_action + noise, -1.0, 1.0)

            return selected_action

    def step(self,
             action: np.ndarray
             ) -> typing.Tuple[np.ndarray, np.float64, bool, dict]:

        next_state, reward, done, info = self.env.step(action)

        if not self.cfg.test:
            transition = (self.current_state, action, reward, next_state, done)
            self.memory.add(transition)

        return next_state, reward, done, info

    def pre_train(self):
        pass

    def train(self):

        for self.i_episode in range(1, self.cfg.num_train_episodes + 1):

            t_begin = time.time()

            self.episode_step = 0
            episode_reward = 0
            done = False
            state = self.env.reset()

            while not done:

                action = self.select_action(state)
                next_state, step_reward, done, _ = self.step(action)

                self.total_step += 1
                self.episode_step += 1

                if len(self.memory) >= self.hyper_params.batch_size:
                    for _ in range(self.hyper_params.multiple_update):
                        experience = self.memory.sample()
                        experience = numpy2floattensor(experience, self.learner.device)
                        self.learner.update_model(experience)

                state = next_state
                episode_reward += step_reward

            t_end = time.time()
            avg_time_cost = (t_end - t_begin) / self.episode_step

            if self.cfg.log_wb or self.cfg.log_cmd:
                time_remaining, quantity_remaining, _, _ = state
                if self.cfg.log_wb:
                    wandb.log(
                        {
                            "episode": self.i_episode,
                            "num steps": self.episode_step,
                            "reward": episode_reward,
                            "time remaining": time_remaining,
                            "quantity remaining": quantity_remaining,
                            "total num steps": self.total_step,
                            "avg time per step": avg_time_cost,
                            "time per episode": t_end - t_begin
                        }
                    )

            if self.i_episode % self.cfg.save_params_every == 0:
                self.learner.save_params(self.i_episode)

        # termination
        self.learner.save_params(self.i_episode)

    def test(self):
        super().test()
