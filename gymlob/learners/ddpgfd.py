import typing
from typing import Union

import gym
import omegaconf
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from gymlob.agents.ddpg import DDPGLearner
from gymlob.learners.learner import TensorTuple


class DDPGfDLearner(DDPGLearner):

    def __init__(self,
                 env: Union[gym.Env, str],
                 cfg: omegaconf.dictconfig.DictConfig,
                 observation_space: gym.spaces.Box,
                 action_space: gym.spaces.Tuple
                 ):

        DDPGLearner.__init__(self, env, cfg, observation_space, action_space)

    def _get_critic_loss(self,
                         experiences: typing.Tuple[TensorTuple, ...],
                         gamma: float
                         ) -> torch.Tensor:
        """Return element-wise critic loss."""
        states, actions, rewards, next_states, dones = experiences[:5]

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        masks = 1 - dones
        next_actions = self.actor_target(next_states)
        next_states_actions = torch.cat((next_states, next_actions), dim=-1)
        next_values = self.critic_target(next_states_actions)
        curr_returns = rewards + gamma * next_values * masks
        curr_returns = curr_returns.to(self.device).detach()

        # train critic
        values = self.critic(torch.cat((states, actions), dim=-1))
        critic_loss_element_wise = (values - curr_returns).pow(2)

        return critic_loss_element_wise

    def update_model(self,
                     experience: Union[TensorTuple, typing.Tuple[TensorTuple]]
                     ) -> TensorTuple:
        """Train the model after each episode."""

        def soft_update(local: nn.Module, target: nn.Module, tau: float):
            """Soft-update: target = tau*local + (1-tau)*target."""
            for t_param, l_param in zip(target.parameters(), local.parameters()):
                t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        use_n_step = self.hyper_params.n_step > 1
        if use_n_step:
            experience_1, experience_n = experience
        else:
            experience_1 = experience

        states, actions = experience_1[:2]
        weights, indices, eps_d = experience_1[-3:]
        gamma = self.hyper_params.gamma

        # train critic
        gradient_clip_ac = self.hyper_params.gradient_clip_ac
        gradient_clip_cr = self.hyper_params.gradient_clip_cr

        critic_loss_element_wise = self._get_critic_loss(experience_1, gamma)
        critic_loss = torch.mean(critic_loss_element_wise * weights)

        if use_n_step:
            gamma = gamma ** self.hyper_params.n_step

            critic_loss_n_element_wise = self._get_critic_loss(experience_n, gamma)
            # to update loss and priorities
            critic_loss_element_wise += (critic_loss_n_element_wise * self.hyper_params.lambda1)
            critic_loss = torch.mean(critic_loss_element_wise * weights)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic.parameters(), gradient_clip_cr)
        self.critic_optim.step()

        # train actor
        actions = self.actor(states)
        actor_loss_element_wise = -self.critic(torch.cat((states, actions), dim=-1))
        actor_loss = torch.mean(actor_loss_element_wise * weights)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(self.actor.parameters(), gradient_clip_ac)
        self.actor_optim.step()

        # update target networks
        soft_update(self.actor, self.actor_target, self.hyper_params.tau)
        soft_update(self.critic, self.critic_target, self.hyper_params.tau)

        # update priorities
        new_priorities = critic_loss_element_wise
        new_priorities += self.hyper_params.lambda3 * actor_loss_element_wise.pow(2)
        new_priorities += self.hyper_params.per_eps
        new_priorities = new_priorities.data.cpu().numpy().squeeze()
        new_priorities += eps_d

        return (actor_loss.item(), critic_loss.item(), indices, new_priorities,)
