import typing
from collections import OrderedDict
from copy import deepcopy

import gym
import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from gymlob.learners.learner import Learner, TensorTuple
from gymlob.utils.nn.neural_network import NeuralNetwork


class DQNLearner(Learner):

    def __init__(self,
                 env: typing.Union[gym.Env, str],
                 cfg: omegaconf.dictconfig.DictConfig,
                 observation_space: gym.spaces.Box,
                 action_space: gym.spaces.Tuple
                 ):

        Learner.__init__(self, env, cfg, observation_space, action_space)

        self.hyper_params = self.cfg.algo.hyper_params
        self.learner_cfg = self.cfg.algo.learner_cfg
        self.use_n_step = self.hyper_params.n_step > 1

        self._init_network()

    def _init_network(self):
        """Initialize networks and optimizers."""

        self.learner_cfg.head.configs.state_size = self.observation_space.shape
        self.learner_cfg.head.configs.output_size = self.action_space.n

        self.dqn = NeuralNetwork(self.learner_cfg.backbone, self.learner_cfg.head).to(self.device)
        self.dqn_target = NeuralNetwork(self.learner_cfg.backbone, self.learner_cfg.head).to(self.device)
        self.loss_fn = DQNLoss()

        self.dqn_target.load_state_dict(self.dqn.state_dict())

        # create optimizer
        self.dqn_optim = optim.Adam(list(self.dqn.parameters()),
                                    lr=self.learner_cfg.optim_cfg.lr_dqn,
                                    weight_decay=self.learner_cfg.optim_cfg.weight_decay,
                                    eps=self.learner_cfg.optim_cfg.adam_eps,)

        # load the optimizer and model parameters
        if self.cfg.load_params_from is not None:
            self.load_params(self.cfg.load_params_from)

    def save_params(self, n_episode: int):

        params = {
            "dqn_state_dict": self.dqn.state_dict(),
            "dqn_target_state_dict": self.dqn_target.state_dict(),
            "dqn_optim_state_dict": self.dqn_optim.state_dict(),
        }

        Learner.save_params(self, params, n_episode)

    def load_params(self, path: str):

        Learner.load_params(self, path)

        params = torch.load(path)
        self.dqn.load_state_dict(params["dqn_state_dict"])
        self.dqn_target.load_state_dict(params["dqn_target_state_dict"])
        self.dqn_optim.load_state_dict(params["dqn_optim_state_dict"])
        print("[INFO] loaded the model and optimizer from", path)

    def update_model(self,
                     experience: typing.Union[TensorTuple, typing.Tuple[TensorTuple]]
                     ) -> typing.Tuple[torch.Tensor, torch.Tensor, list, np.ndarray]:
        """Update dqn and dqn target."""

        def soft_update(local: nn.Module, target: nn.Module, tau: float):
            """Soft-update: target = tau*local + (1-tau)*target."""
            for t_param, l_param in zip(target.parameters(), local.parameters()):
                t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        if self.use_n_step:
            experience_1, experience_n = experience
        else:
            experience_1 = experience

        weights, indices = experience_1[-3:-1]

        gamma = self.hyper_params.gamma

        dq_loss_element_wise, q_values = self.loss_fn(self.dqn, self.dqn_target, experience_1, gamma)

        dq_loss = torch.mean(dq_loss_element_wise * weights)

        # n step loss
        if self.use_n_step:
            gamma = self.hyper_params.gamma ** self.hyper_params.n_step

            dq_loss_n_element_wise, q_values_n = self.loss_fn(self.dqn, self.dqn_target, experience_n, gamma)

            # to update loss and priorities
            q_values = 0.5 * (q_values + q_values_n)
            dq_loss_element_wise += dq_loss_n_element_wise * self.hyper_params.w_n_step
            dq_loss = torch.mean(dq_loss_element_wise * weights)

        # q_value regularization
        q_regular = torch.norm(q_values, 2).mean() * self.hyper_params.w_q_reg

        # total loss
        loss = dq_loss + q_regular

        self.dqn_optim.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), self.hyper_params.gradient_clip)
        self.dqn_optim.step()

        # update target networks
        soft_update(self.dqn, self.dqn_target, self.hyper_params.tau)

        # update priorities in PER
        loss_for_prior = dq_loss_element_wise.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.hyper_params.per_eps

        if self.learner_cfg.head.configs.use_noisy_net:
            self.dqn.head.reset_noise()
            self.dqn_target.head.reset_noise()

        return (loss.item(), q_values.mean().item(), indices, new_priorities,)

    def get_state_dict(self) -> OrderedDict:
        """Return state dicts, mainly for distributed worker."""
        dqn = deepcopy(self.dqn)
        return dqn.cpu().state_dict()

    def get_policy(self) -> nn.Module:
        """Return model (policy) used for action selection, used only in grad cam."""
        return self.dqn


class DQNLoss:

    def __call__(
        self,
        model: NeuralNetwork,
        target_model: NeuralNetwork,
        experiences: typing.Tuple[torch.Tensor, ...],
        gamma: float
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Return element-wise dqn loss and Q-values."""
        states, actions, rewards, next_states, dones = experiences[:5]

        q_values = model(states)
        # According to noisynet paper,
        # it resamples noisynet parameters on online network when using double q
        # but we don't because there is no remarkable difference in performance.
        next_q_values = model(next_states)

        next_target_q_values = target_model(next_states)

        curr_q_value = q_values.gather(1, actions.long().unsqueeze(1))
        next_q_value = next_target_q_values.gather(1, next_q_values.argmax(1).unsqueeze(1))

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        masks = 1 - dones
        target = rewards + gamma * next_q_value * masks

        # calculate dq loss
        dq_loss_element_wise = F.smooth_l1_loss(curr_q_value, target.detach(), reduction="none")

        return dq_loss_element_wise, q_values