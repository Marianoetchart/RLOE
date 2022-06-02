import time
import typing

import gym
import numpy as np
import omegaconf
import torch
import wandb

from gymlob.agents.agent import Agent
from gymlob.learners.dqn import DQNLearner

from gymlob.utils.replay_buffer import ReplayBuffer, PrioritizedBufferWrapper
from gymlob.utils.utils import numpy2floattensor, add_widxheight_dim


class DQNAgent(Agent):

    def __init__(self,
                 env: gym.Env,
                 cfg: omegaconf.dictconfig.DictConfig
                 ):

        Agent.__init__(self, env, cfg)

        self.learner = DQNLearner(self.env, cfg, self.observation_space, self.action_space)

        self.per_beta = self.hyper_params.per_beta
        self.use_n_step = self.hyper_params.n_step > 1
        self.epsilon = self.hyper_params.max_epsilon

        if not self.cfg.test:
            # replay memory for a single step
            self.memory = ReplayBuffer(self.hyper_params.buffer_size, self.hyper_params.batch_size)
            self.memory = PrioritizedBufferWrapper(self.memory, alpha=self.hyper_params.per_alpha)

            # replay memory for multi-steps
            if self.use_n_step:
                self.memory_n = ReplayBuffer(self.hyper_params.buffer_size, self.hyper_params.batch_size,
                                             n_step=self.hyper_params.n_step, gamma=self.hyper_params.gamma)

        self.current_state = np.zeros(1)
        self.episode_step = 0
        self.i_episode = 0

    def select_action(self,
                      state: np.ndarray
                      ) -> np.ndarray:

        self.current_state = state

        # epsilon greedy policy
        if not self.cfg.test and self.epsilon > np.random.random():
            selected_action = np.array(self.env.action_space.sample())
        else:
            with torch.no_grad():
                state = numpy2floattensor(state, self.learner.device)
                selected_action = self.learner.dqn(add_widxheight_dim(state)).argmax()
            selected_action = selected_action.detach().cpu().numpy()
        return selected_action

    def step(self,
             action: np.ndarray
             ) -> typing.Tuple[np.ndarray, np.float64, bool, dict]:

        next_state, reward, done, info = self.env.step(action)

        if np.isnan(reward).any():
            print('Error')

        if not self.cfg.test:
            transition = (self.current_state, action, reward, next_state, done)

            if self.use_n_step:
                transition = self.memory_n.add(transition)

            if transition:
                self.memory.add(transition)

        return next_state, reward, done, info

    def pre_train(self):
        pass

    def sample_experience(self) -> typing.Tuple[torch.Tensor, ...]:
        """Sample experience from replay buffer."""
        experiences_1 = self.memory.sample(self.per_beta)
        experiences_1 = (numpy2floattensor(experiences_1[:6], self.learner.device) + experiences_1[6:])

        if self.use_n_step:
            indices = experiences_1[-2]
            experiences_n = self.memory_n.sample(indices)
            return (experiences_1, numpy2floattensor(experiences_n, self.learner.device),)

        return experiences_1

    def train(self):

        for self.i_episode in range(1, self.cfg.num_train_episodes + 1):

            t_begin = time.time()

            self.episode_step = 0
            episode_reward = 0
            done = False
            state = self.env.reset()

            while not done:

                action = self.select_action(state)
                next_state, reward, done, state_info = self.step(action)

                self.total_step += 1
                self.episode_step += 1

                if len(self.memory) >= self.hyper_params.update_starts_from:
                    if self.total_step % self.hyper_params.train_freq == 0:
                        for _ in range(self.hyper_params.multiple_update):
                            experience = self.sample_experience()
                            info = self.learner.update_model(experience)
                            indices, new_priorities = info[2:4]
                            self.memory.update_priorities(indices, new_priorities)

                    # decrease epsilon
                    self.epsilon = max(self.epsilon
                                       - (self.hyper_params.max_epsilon - self.hyper_params.min_epsilon)
                                       * self.hyper_params.epsilon_decay,
                                       self.hyper_params.min_epsilon)

                    # increase priority beta
                    fraction = min(float(self.i_episode) / self.cfg.num_train_episodes, 1.0)
                    self.per_beta = self.per_beta + fraction * (1.0 - self.per_beta)

                state = next_state
                episode_reward += reward

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
                            "time per episode": t_end - t_begin,
                            "implementation shortfall" :  state_info.implementation_shortfall
                        }
                    )

            if self.i_episode % self.cfg.save_params_every == 0:
                self.learner.save_params(self.i_episode)

        # termination
        self.learner.save_params(self.i_episode)

    def test(self):
        super().test()
