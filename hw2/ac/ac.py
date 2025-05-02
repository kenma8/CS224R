import hydra
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class Actor(nn.Module):
    def __init__(self, obs_shape, action_shape, hidden_dim, std=0.1):
        super().__init__()

        self.std = std
        self.policy = nn.Sequential(nn.Linear(obs_shape[0], hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs):
        mu = self.policy(obs)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * self.std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, obs_shape, action_shape, num_critics,
                 hidden_dim):
        super().__init__()

        self.critics = nn.ModuleList([nn.Sequential(
            nn.Linear(obs_shape[0] + action_shape[0], hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
            for _ in range(num_critics)])

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h_action = torch.cat([obs, action], dim=-1)
        return [critic(h_action) for critic in self.critics]


class ACAgent:
    def __init__(self, obs_shape, action_shape, device, lr,
                 hidden_dim, num_critics, critic_target_tau, stddev_clip, use_tb):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.use_tb = use_tb
        self.stddev_clip = stddev_clip

        # models
        self.actor = Actor(obs_shape, action_shape,
                           hidden_dim).to(device)

        self.critic = Critic(obs_shape, action_shape,
                             num_critics, hidden_dim).to(device)
        self.critic_target = Critic(obs_shape, action_shape,
                                    num_critics, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        dist = self.actor(obs.unsqueeze(0))
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
        return action.cpu().numpy()[0]

    def update_critic(self, replay_iter):
        '''
        This function updates the critic and target critic parameters.

        Args:

        replay_iter:
            An iterable that produces batches of tuples
            (observation, action, reward, discount, next_observation),
            where:
            observation: array of shape [batch, D] of states
            action: array of shape [batch, action_dim]
            reward: array of shape [batch,]
            discount: array of shape [batch,]
            next_observation: array of shape [batch, D] of states

        Returns:

        metrics: dictionary of relevant metrics to be logged. Add any metrics
                 that you find helpful to log for debugging, such as the critic
                 loss, or the mean Bellman targets.
        '''

        metrics = dict()

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        ### YOUR CODE HERE ###

        # Sample next state actions from policy
        with torch.no_grad():
            next_actions_dist = self.actor(next_obs)
            next_actions = next_actions_dist.sample(clip=self.stddev_clip)
            h_action = torch.cat([next_obs, next_actions], dim=-1)
        
        # Compute Bellman targets
        random_critics = random.sample(list(self.critic_target.critics), 2)
        y = reward + discount * torch.min(random_critics[0](h_action),
                                          random_critics[1](h_action))
        
        # Compute the loss
        vec_loss = 0
        y = y.detach()
        preds = self.critic.forward(obs, action)
        for pred in preds:
            vec_loss += torch.pow((pred - y), 2)
        loss = torch.mean(vec_loss)
        metrics['loss'] = loss.item()

        # Take a gradient step with respect to the critic parameters
        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()

        # Update the target critic parameters using exponential moving average
        for critic, target_critic in zip(self.critic.critics, self.critic_target.critics):
            critic_params = critic.parameters()
            target_critic_params = target_critic.parameters()
            for param, target_param in zip(critic_params, target_critic_params):
                target_param.data.copy_(self.critic_target_tau * param.data + (1 - self.critic_target_tau) * target_param.data)

        #####################
        return metrics

    def update_actor(self, replay_iter):
        '''
        This function updates the policy parameters.

        Args:

        replay_iter:
            An iterable that produces batches of tuples
            (observation, action, reward, discount, next_observation),
            where:
            observation: array of shape [batch, D] of states
            action: array of shape [batch, action_dim]
            reward: array of shape [batch,]
            discount: array of shape [batch,]
            next_observation: array of shape [batch, D] of states

        Returns:

        metrics: dictionary of relevant metrics to be logged. Add any metrics
                 that you find helpful to log for debugging, such as the actor
                 loss.
        '''
        metrics = dict()

        batch = next(replay_iter)
        obs, _, _, _, _ = utils.to_torch(
            batch, self.device)

        ### YOUR CODE HERE ###
        
        # Sample actions from the actor
        actions_dist = self.actor(obs)
        actions = actions_dist.sample(clip=self.stddev_clip)

        # Compute the objective that optimizes the actor to maximize the Q-value estimates from the critics
        critic_values = self.critic.forward(obs, actions)
        stacked_critic_values = torch.stack(critic_values)
        loss = -stacked_critic_values.mean()
        metrics['loss'] = loss.item()

        # Take a gradient step on this objective with respect to the policy only
        self.actor_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()

        return metrics

    def bc(self, replay_iter):
        '''
        This function updates the policy with end-to-end
        behavior cloning

        Args:

        replay_iter:
            An iterable that produces batches of tuples
            (observation, action, reward, discount, next_observation),
            where:
            observation: array of shape [batch, D] of states
            action: array of shape [batch, action_dim]
            reward: array of shape [batch,]
            discount: array of shape [batch,]
            next_observation: array of shape [batch, D] of states

        Returns:

        metrics: dictionary of relevant metrics to be logged. Add any metrics
                 that you find helpful to log for debugging, such as the loss.
        '''

        metrics = dict()

        batch = next(replay_iter)
        obs, action, _, _, _ = utils.to_torch(batch, self.device)

        ### YOUR CODE HERE ###
        pred_actions_dist = self.actor(obs)
        loss = -pred_actions_dist.log_prob(action).mean()
        self.actor_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()
        metrics['loss'] = loss.item()

        return metrics
