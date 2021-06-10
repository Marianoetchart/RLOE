import time
import typing
import logging as log

import gym
import numpy as np
import omegaconf
import pickle
import torch
import wandb

from gymlob.agents.ddpg import DDPGAgent
from gymlob.learners.ddpgfd import DDPGfDLearner

from gymlob.utils.replay_buffer import ReplayBuffer, PrioritizedBufferWrapper, get_n_step_info_from_demo
from gymlob.utils.utils import numpy2floattensor


class DDPGfDAgent(DDPGAgent):

    def __init__(self,
                 env: gym.Env,
                 cfg: omegaconf.dictconfig.DictConfig
                 ):

        DDPGAgent.__init__(self, env, cfg)

        self.learner = DDPGfDLearner(env, cfg, self.observation_space, self.action_space)

        self.per_beta = self.hyper_params.per_beta
        self.use_n_step = self.hyper_params.n_step > 1

        if not self.cfg.test:

            with open(self.cfg.demo_path, "rb") as f:
                demos = pickle.load(f)

            if self.use_n_step:
                demos, demos_n_step = get_n_step_info_from_demo(demos,
                                                                self.hyper_params.n_step,
                                                                self.hyper_params.gamma)

                # replay memory for multi-steps
                self.memory_n = ReplayBuffer(max_len=self.hyper_params.buffer_size,
                                             batch_size=self.hyper_params.batch_size,
                                             n_step=self.hyper_params.n_step,
                                             gamma=self.hyper_params.gamma,
                                             demo=demos_n_step)

            # replay memory for a single step
            self.memory = ReplayBuffer(self.hyper_params.buffer_size,
                                       self.hyper_params.batch_size,
                                       demo=demos)
            self.memory = PrioritizedBufferWrapper(self.memory,
                                                   alpha=self.hyper_params.per_alpha,
                                                   epsilon_d=self.hyper_params.per_eps_demo)

    def sample_experience(self) -> typing.Tuple[torch.Tensor, ...]:

        experiences_1 = self.memory.sample(self.per_beta)
        experiences_1 = (numpy2floattensor(experiences_1[:6], self.learner.device) + experiences_1[6:])
        if self.use_n_step:
            indices = experiences_1[-2]
            experiences_n = self.memory_n.sample(indices)
            return (experiences_1, numpy2floattensor(experiences_n, self.learner.device),)
        return experiences_1

    def pre_train(self):

        pretrain_loss = list()
        pretrain_step = self.hyper_params.pretrain_step
        log.info("Pre-Train {} step.".format(pretrain_step))

        for i_step in range(1, pretrain_step + 1):
            experience = self.sample_experience()
            info = self.learner.update_model(experience)
            loss = info[0:2]
            pretrain_loss.append(loss)  # for logging

            if self.cfg.log and i_step % 100 == 0:
                avg_reward = self.interim_test(num_rollout_episodes=20)
                avg_loss = np.vstack(pretrain_loss).mean(axis=0)
                wandb.log(
                    {
                        "offline step": i_step,
                        "offline step rollout avg total reward": avg_reward,
                        "offline step total avg loss": avg_loss.sum()
                    }
                )
                pretrain_loss.clear()

        log.info("Pre-Train Complete!\n")
        log.info("Saving Pre-trained model!")
        self.learner.save_params(n_episode=0)

    def train(self):

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

                if len(self.memory) >= self.hyper_params.batch_size:
                    for _ in range(self.hyper_params.multiple_update):
                        experience = self.sample_experience()
                        info = self.learner.update_model(experience)
                        loss = info[0:2]
                        indices, new_priorities = info[2:4]
                        losses.append(loss)  # for logging
                        self.memory.update_priorities(indices, new_priorities)

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

    def test(self):
        super().test()