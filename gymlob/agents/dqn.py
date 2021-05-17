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
from gymlob.utils.utils import numpy2floattensor


class DQNAgent(Agent):

    def __init__(self,
                 env: typing.Union[gym.Env, str],
                 cfg: omegaconf.dictconfig.DictConfig
                 ):

        Agent.__init__(self, env, cfg)

        self.curr_state = np.zeros(1)
        self.episode_step = 0
        self.i_episode = 0

        self.per_beta = self.hyper_params.per_beta
        self.use_n_step = self.hyper_params.n_step > 1

        if self.learner_cfg.head.configs.use_noisy_net:
            self.max_epsilon = 0.0
            self.min_epsilon = 0.0
            self.epsilon = 0.0
        else:
            self.max_epsilon = self.hyper_params.max_epsilon
            self.min_epsilon = self.hyper_params.min_epsilon
            self.epsilon = self.hyper_params.max_epsilon

        if not self.cfg.test:
            # replay memory for a single step
            self.memory = ReplayBuffer(self.hyper_params.buffer_size, self.hyper_params.batch_size)
            self.memory = PrioritizedBufferWrapper(self.memory, alpha=self.hyper_params.per_alpha)

            # replay memory for multi-steps
            if self.use_n_step:
                self.memory_n = ReplayBuffer(self.hyper_params.buffer_size, self.hyper_params.batch_size,
                                             n_step=self.hyper_params.n_step, gamma=self.hyper_params.gamma)

        self.learner = DQNLearner(self.env, cfg, self.observation_space, self.action_space)

    def select_action(self,
                      state: np.ndarray
                      ) -> np.ndarray:
        """Select an action from the input space."""
        self.curr_state = state

        # epsilon greedy policy
        if not self.cfg.test and self.epsilon > np.random.random():
            selected_action = np.array(self.env.action_space.sample())
        else:
            with torch.no_grad():
                state = numpy2floattensor(state, self.learner.device)
                selected_action = self.learner.dqn(state).argmax()
            selected_action = selected_action.detach().cpu().numpy()
        return selected_action

    def step(self,
             action: np.ndarray
             ) -> typing.Tuple[np.ndarray, np.float64, bool, dict]:
        """Take an action and return the response of the env."""
        next_state, reward, done, info = self.env.step(action)

        if not self.cfg.test:
            # if the last state is not a terminal state, store done as false
            done_bool = False if self.episode_step == self.cfg.max_episode_steps else done

            transition = (self.curr_state, action, reward, next_state, done_bool)

            # add n-step transition
            if self.use_n_step:
                transition = self.memory_n.add(transition)

            # add a single step transition
            # if transition is not an empty tuple
            if transition:
                self.memory.add(transition)

        return next_state, reward, done, info

    def pre_train(self):
        """

        :return:
        """
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

        # pre-training if needed
        self.pre_train()

        for self.i_episode in range(1, self.cfg.num_episodes + 1):

            state = self.env.reset()
            done = False
            score = 0
            self.episode_step = 0
            losses = list()

            t_begin = time.time()

            while not done:

                action = self.select_action(state)
                next_state, reward, done, _ = self.step(action)
                self.total_step += 1
                self.episode_step += 1

                if len(self.memory) >= self.hyper_params.update_starts_from:
                    if self.total_step % self.hyper_params.train_freq == 0:
                        for _ in range(self.hyper_params.multiple_update):
                            experience = self.sample_experience()
                            info = self.learner.update_model(experience)
                            loss = info[0:2]
                            indices, new_priorities = info[2:4]
                            losses.append(loss)  # for logging
                            self.memory.update_priorities(indices, new_priorities)

                    # decrease epsilon
                    self.epsilon = max(self.epsilon
                        - (self.max_epsilon - self.min_epsilon)
                        * self.hyper_params.epsilon_decay,
                        self.min_epsilon)

                    # increase priority beta
                    fraction = min(float(self.i_episode) / self.cfg.num_episodes, 1.0)
                    self.per_beta = self.per_beta + fraction * (1.0 - self.per_beta)

                state = next_state
                score += reward

            t_end = time.time()
            avg_time_cost = (t_end - t_begin) / self.episode_step

            # logging
            if losses and self.cfg.log:
                avg_loss = np.vstack(losses).mean(axis=0)
                wandb.log(
                    {
                        "episode": self.i_episode,
                        "episode num steps": self.episode_step,
                        "total num steps": self.total_step,
                        "episode total reward": score,
                        "episode actor avg loss": avg_loss[0],
                        "episode critic avg loss": avg_loss[1],
                        "episode total avg loss": avg_loss.sum(),
                        "episode avg time per step": avg_time_cost
                    }
                )
                losses.clear()

            if self.i_episode % self.cfg.save_params_every == 0:
                self.learner.save_params(self.i_episode)

        # termination
        self.env.close()
        self.learner.save_params(self.i_episode)