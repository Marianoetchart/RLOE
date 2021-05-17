import gym
import omegaconf
import typing
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from gymlob.learners.learner import Learner
from gymlob.utils.nn.neural_network import NeuralNetwork


class DDPGLearner(Learner):

    def __init__(self,
                 env: typing.Union[gym.Env, str],
                 cfg: omegaconf.dictconfig.DictConfig,
                 observation_space: gym.spaces.Box,
                 action_space: gym.spaces.Tuple):

        Learner.__init__(self, env, cfg, observation_space, action_space)

        self.hyper_params = self.cfg.algo.hyper_params
        self.learner_cfg = self.cfg.algo.learner_cfg

        self.learner_cfg.head.actor.configs.state_size = self.observation_space.shape
        self.learner_cfg.head.actor.configs.output_size = self.action_space.shape[0]
        self.learner_cfg.head.critic.configs.state_size = (self.observation_space.shape[0] +
                                                           self.action_space.shape[0],)
        self._init_network()

    def _init_network(self):

        # create actor

        self.actor = NeuralNetwork(self.learner_cfg.backbone.actor,
                                   self.learner_cfg.head.actor).to(self.device)
        self.actor_target = NeuralNetwork(self.learner_cfg.backbone.actor,
                                          self.learner_cfg.head.actor).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # create critic
        self.critic = NeuralNetwork(self.learner_cfg.backbone.critic,
                                    self.learner_cfg.head.critic).to(self.device)
        self.critic_target = NeuralNetwork(self.learner_cfg.backbone.critic,
                                           self.learner_cfg.head.critic).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # create optimizer
        self.actor_optim = optim.Adam(params=list(self.actor.parameters()),
                                      lr=self.learner_cfg.optim_cfg.lr_actor,
                                      weight_decay=self.learner_cfg.optim_cfg.weight_decay)

        self.critic_optim = optim.Adam(params=list(self.critic.parameters()),
                                       lr=self.learner_cfg.optim_cfg.lr_critic,
                                       weight_decay=self.learner_cfg.optim_cfg.weight_decay)

        # load the optimizer and model parameters
        if self.cfg.load_params_from is not None:
            self.load_params(self.cfg.load_params_from)

    def save_params(self, n_episode: int):

        params = {
            "actor_state_dict": self.actor.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optim_state_dict": self.actor_optim.state_dict(),
            "critic_optim_state_dict": self.critic_optim.state_dict(),
        }
        Learner.save_params(self, params, n_episode)

    def load_params(self, path: str):

        Learner.load_params(self, path)

        params = torch.load(path)
        self.actor.load_state_dict(params["actor_state_dict"])
        self.actor_target.load_state_dict(params["actor_target_state_dict"])
        self.critic.load_state_dict(params["critic_state_dict"])
        self.critic_target.load_state_dict(params["critic_target_state_dict"])
        self.actor_optim.load_state_dict(params["actor_optim_state_dict"])
        self.critic_optim.load_state_dict(params["critic_optim_state_dict"])
        print("[INFO] loaded the model and optimizer from", path)

    def update_model(self, experience: typing.Tuple[torch.Tensor, ...]) -> typing.Tuple[torch.Tensor, ...]:

        def soft_update(local: nn.Module, target: nn.Module, tau: float):
            """Soft-update: target = tau*local + (1-tau)*target."""
            for t_param, l_param in zip(target.parameters(), local.parameters()):
                t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        states, actions, rewards, next_states, dones = experience

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        masks = 1 - dones
        next_actions = self.actor_target(next_states)
        next_values = self.critic_target(torch.cat((next_states, next_actions), dim=-1))
        curr_returns = rewards + self.hyper_params.gamma * next_values * masks
        curr_returns = curr_returns.to(self.device)

        # train critic
        gradient_clip_ac = self.hyper_params.gradient_clip_ac
        gradient_clip_cr = self.hyper_params.gradient_clip_cr

        values = self.critic(torch.cat((states, actions), dim=-1))
        critic_loss = F.mse_loss(values, curr_returns)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic.parameters(), gradient_clip_cr)
        self.critic_optim.step()

        # train actor
        actions = self.actor(states)
        actor_loss = -self.critic(torch.cat((states, actions), dim=-1)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(self.actor.parameters(), gradient_clip_ac)
        self.actor_optim.step()

        # update target networks
        soft_update(self.actor, self.actor_target, self.hyper_params.tau)
        soft_update(self.critic, self.critic_target, self.hyper_params.tau)

        return actor_loss.item(), critic_loss.item()

    def get_state_dict(self) -> typing.Tuple[OrderedDict]:
        return (self.critic_target.state_dict(), self.actor.state_dict())

    def get_policy(self) -> nn.Module:
        return self.actor