import abc
import typing
import os
import omegaconf
import wandb

import gym
import pandas as pd
import numpy as np
import torch


class Agent(abc.ABC):

    def __init__(self,
                 env: gym.Env,
                 cfg: omegaconf.dictconfig.DictConfig
                 ):
        self.env = env
        self.cfg = cfg

        self.hyper_params = self.cfg.algo.hyper_params
        self.learner_cfg = self.cfg.algo.learner_cfg

        self.observation_space = self.get_observation_space()
        self.action_space = self.get_action_space()

        self.total_step = 0
        self.learner = None

    def get_observation_space(self):
        return self.env.observation_space

    def get_action_space(self):
        return self.env.action_space

    @abc.abstractmethod
    def select_action(self,
                      state: np.ndarray
                      ) -> typing.Union[torch.Tensor, np.ndarray]:
        pass

    @abc.abstractmethod
    def step(self,
             action: typing.Union[torch.Tensor, np.ndarray]
             ) -> typing.Tuple[np.ndarray, np.float64, bool, dict]:
        pass

    @abc.abstractmethod
    def pre_train(self):
        pass

    @abc.abstractmethod
    def train(self):
        pass

    def test(self):

        def save_transitions():
            #TODO: change this
            demos_folder_path = f"'../../../../../data/{self.cfg.project_name}/{self.cfg.algo.env}/{self.cfg.algo.seed}/"
            demos_file_path = f"{demos_folder_path}{self.cfg.experiment_name}_demos.pkl"
            print(f"Saving {len(transitions)} to {demos_file_path}")
            if os.path.isfile(demos_file_path):
                existing_demos = pd.read_pickle(demos_file_path)
                pd.to_pickle(existing_demos + transitions, demos_file_path)
            else:
                os.makedirs(demos_folder_path, exist_ok=True)
                pd.to_pickle(transitions, demos_file_path)

        transitions = []
        for i_episode in range(self.cfg.num_test_episodes):

            state = self.env.reset()
            done = False
            score = 0
            step = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = self.step(action)

                transition = (state, action, reward, next_state, done)
                transitions.append(transition)

                state = next_state
                score += reward
                step += 1

            if self.cfg.log:
                wandb.log(
                    {
                        "test episode": i_episode,
                        "test episode num steps": step,
                        "test episode total reward": score,
                    }
                )

        if self.cfg.save_test_episodes:
            save_transitions()

    def interim_test(self,
                     num_rollout_episodes: int
                     ) -> float:

        rollout_rewards = list()
        for i_episode in range(num_rollout_episodes):
            state = self.env.reset()
            done = False
            score = 0
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = self.step(action)
                state = next_state
                score += reward
            rollout_rewards.append(score)

        return sum(rollout_rewards) / len(rollout_rewards)