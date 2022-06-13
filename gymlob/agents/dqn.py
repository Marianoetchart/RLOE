import time
import statistics
import typing

import gym
import numpy as np
import pandas as pd 
import os
import omegaconf
import torch
import wandb

from gymlob.agents.agent import Agent
from gymlob.learners.dqn import DQNLearner

from gymlob.utils.replay_buffer import ReplayBuffer, PrioritizedBufferWrapper
from gymlob.utils.utils import numpy2floattensor, add_widxheight_dim, LinearSchedule


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
        self.ep_schedule = LinearSchedule(self.hyper_params.epsilon_timesteps,self.hyper_params.min_epsilon,self.hyper_params.max_epsilon )

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
        self.exploitation_steps = 0
        self.i_episode = 0

    def select_action(self,
                      state: np.ndarray
                      ) -> np.ndarray:

        self.current_state = state
        time_remaining, quantity_remaining, spread, order_volume_imbalance, action = self.current_state
        
        # epsilon greedy policy
        if not self.cfg.test and (self.epsilon > np.random.random()):
            #if a binomial distribution across the action is selected, select actions that average to TWAP policy
            if self.cfg.algo.hyper_params.use_binomial_egreedy:
                selected_action = np.array(np.random.binomial(quantity_remaining, 1/(time_remaining)))
            else:
                selected_action = np.array( np.random.randint(0,quantity_remaining))
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
            running_loss = 0
            implementation_shortfalls = []
            perm_impacts= [] 
            temp_impacts= [] 
            slippages = [] 
            done = False
            state = self.env.reset()

            while not done:

                action = self.select_action(state)
                next_state, reward, done, state_info = self.step(action)

                self.total_step += 1
                self.episode_step += 1
                state = next_state
                episode_reward += reward
                implementation_shortfalls.append(state_info.implementation_shortfall)
                perm_impacts.append(state_info.currentPermanentImpact)
                temp_impacts.append(state_info.currentTemporaryImpact)
                slippages.append(state_info.slippage)


                if len(self.memory) >= self.hyper_params.update_starts_from:
                    if self.total_step % self.hyper_params.train_freq == 0:
                        for _ in range(self.hyper_params.multiple_update):
                            experience = self.sample_experience()
                            loss, q_values, indices, new_priorities = self.learner.update_model(experience)
                            self.memory.update_priorities(indices, new_priorities)
                            running_loss += loss
                    #self.epsilon = max(self.epsilon
                    #                   - (self.hyper_params.max_epsilon - self.hyper_params.min_epsilon)
                    #                   * self.hyper_params.epsilon_decay,
                    #                   self.hyper_params.min_epsilon)

                    # increase priority beta
                    fraction = min(float(self.i_episode) / self.cfg.num_train_episodes, 1.0)
                    self.per_beta = self.per_beta + fraction * (1.0 - self.per_beta)
                
                # decrease epsilon
                self.epsilon = self.ep_schedule.value(self.i_episode)

            t_end = time.time()
            avg_time_cost = (t_end - t_begin) / self.episode_step

            if self.cfg.log_wb or self.cfg.log_cmd:
                time_remaining, quantity_remaining, _, _, action = state
                if self.cfg.log_wb and (self.i_episode % self.cfg.log_freq == 0):
                    wandb.log(
                        {
                            "episode": self.i_episode,
                            "Num Steps": self.episode_step,
                            "Episode Reward": episode_reward,
                            "Time Remaining": time_remaining,
                            "Quantity Remaining": quantity_remaining,
                            "End Action": action,
                            "Total Num Steps": self.total_step,
                            "Avg Time Per Step": avg_time_cost,
                            "Time Per Episode": t_end - t_begin,
                            "Implementation Shortfall" :  state_info.implementation_shortfall,
                            "Avg Permanent Impact" :  statistics.mean(perm_impacts),
                            "Avg Temporary Impact" :  statistics.mean(temp_impacts),
                            "Avg Slippage": statistics.mean(slippages),
                            "Memory Size": len(self.memory_n),
                        }
                    )
                    if running_loss != 0:
                        wandb.log(
                            {
                                "episode": self.i_episode,
                                "Average Action Values (Q)": q_values,
                                "Training Loss": running_loss
                            }
                        ) 

            if self.i_episode % self.cfg.save_params_every == 0:
                self.learner.save_params(self.i_episode)

        # termination
        self.learner.save_params(self.i_episode)

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
            total_reward = 0
            bm_reward = 0 
            total_capture= 0
            slippages = []
            
            bm_total_capture= 0
            step = 0

            while not done:
                if self.cfg.twap == True:
                    action = self.cfg.quantity / self.cfg.duration 
                else:
                    action = self.select_action(state)
        
                next_state, reward, done, state_info = self.step(action)

                total_capture += state_info.totalCapture
                slippages.append(state_info.slippage)

                transition = (state, action, reward, next_state, done)
                transitions.append(transition)

                state = next_state
                total_reward += reward
                step += 1

            if self.cfg.log_wb:
                wandb.log(
                    {
                        "episode": i_episode,
                        "test episode num steps": step,
                        "test episode total reward": total_reward,
                        "Implementation Shortfall" :  state_info.implementation_shortfall,
                        "P&L": total_capture,
                        "Avg Slippage": statistics.mean(slippages),
                        
                    }
                )

        if self.cfg.save_test_episodes:
            save_transitions()