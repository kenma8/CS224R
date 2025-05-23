from collections import OrderedDict

from cs224r.critics.dqn_critic import DQNCritic
from cs224r.critics.cql_critic import CQLCritic
from cs224r.critics.iql_critic import IQLCritic
from cs224r.infrastructure.replay_buffer import ReplayBuffer
from cs224r.infrastructure.utils import *
from cs224r.infrastructure import pytorch_util as ptu
from cs224r.policies.argmax_policy import ArgMaxPolicy
from cs224r.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer
from cs224r.exploration.rnd_model import RNDModel
from .dqn_agent import DQNAgent
from cs224r.policies.MLP_policy import MLPPolicyAWAC
import numpy as np
import torch


class IQLAgent(DQNAgent):
    def __init__(self, env, agent_params, normalize_rnd=True, rnd_gamma=0.99):
        super(IQLAgent, self).__init__(env, agent_params)
        
        self.replay_buffer = MemoryOptimizedReplayBuffer(100000, 1, float_obs=True)
        self.num_exploration_steps = agent_params['num_exploration_steps']
        self.offline_exploitation = agent_params['offline_exploitation']

        self.exploitation_critic = IQLCritic(agent_params, self.optimizer_spec)
        self.exploration_critic = DQNCritic(agent_params, self.optimizer_spec)
        
        self.exploration_model = RNDModel(agent_params, self.optimizer_spec)
        self.explore_weight_schedule = agent_params['explore_weight_schedule']
        self.exploit_weight_schedule = agent_params['exploit_weight_schedule']
        
        self.use_boltzmann = agent_params['use_boltzmann']
        self.actor = ArgMaxPolicy(self.exploitation_critic)
        self.eval_policy = self.awac_actor = MLPPolicyAWAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
            self.agent_params['awac_lambda'],
        )

        self.exploit_rew_shift = agent_params['exploit_rew_shift']
        self.exploit_rew_scale = agent_params['exploit_rew_scale']
        self.eps = agent_params['eps']

        self.running_rnd_rew_std = 1
        self.normalize_rnd = normalize_rnd
        self.rnd_gamma = rnd_gamma

    def get_qvals(self, critic, obs, action=None, use_v=False):
        if use_v:
            q_value = critic.v_net(obs)
        else:
            qa_values = critic.q_net_target(obs)
            q_value = torch.gather(qa_values, 1, action.type(torch.int64).unsqueeze(1))
        return q_value

    def estimate_advantage(self, ob_no, ac_na, re_n, next_ob_no, terminal_n,
                           n_actions=10):
        # Note: n_actions is not used
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na)
        re_n = ptu.from_numpy(re_n)
        next_ob_no = ptu.from_numpy(next_ob_no)
        terminal_n = ptu.from_numpy(terminal_n)
        # TODO: Estimate the advantage function
        # HINT: Use get_qvals with the appropriate arguments
        # HINT: Access critic using self.exploitation_critic 
        # (critic trained in the offline setting)
        ### YOUR CODE START HERE ###
        q_est = self.get_qvals(self.exploitation_critic, ob_no, action=ac_na).squeeze(1)
        v_est = self.get_qvals(self.exploitation_critic, ob_no, use_v=True).squeeze(1)
        advantage = q_est - v_est
        return advantage
        ### YOUR CODE END HERE ###
        
    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = {}

        if self.t > self.num_exploration_steps:
            self.actor.set_critic(self.exploitation_critic)
            self.actor.use_boltzmann = False

        if (self.t > self.learning_starts
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
        ):

           
            explore_weight = self.explore_weight_schedule.value(self.t)
            exploit_weight = self.exploit_weight_schedule.value(self.t)


            expl_bonus = self.exploration_model.forward_np(ob_no)
            if self.normalize_rnd:
                expl_bonus = normalize(expl_bonus, 0, self.running_rnd_rew_std)
                self.running_rnd_rew_std = (self.rnd_gamma * self.running_rnd_rew_std 
                    + (1 - self.rnd_gamma) * expl_bonus.std())

          
            mixed_reward = explore_weight * expl_bonus + exploit_weight * re_n
            env_reward = (re_n + self.exploit_rew_shift) * self.exploit_rew_scale


            # Update Critics And Exploration Model #
            expl_model_loss = self.exploration_model.update(next_ob_no)
            exploration_critic_loss = self.exploration_critic.update(ob_no, ac_na, next_ob_no, mixed_reward, terminal_n)
            exploitation_critic_loss = self.exploitation_critic.update_v(ob_no, ac_na)
            exploitation_critic_loss.update(self.exploitation_critic.update_q(ob_no, ac_na, next_ob_no, env_reward, terminal_n))


            #update actor
            # TODO 1): Estimate the advantage
            # TODO 2): Calculate the awac actor loss
            
            ### YOUR CODE START HERE ###
            advantage = self.estimate_advantage(ob_no, ac_na, re_n, next_ob_no, terminal_n)
            actor_loss = self.eval_policy.update(ob_no, ac_na, advantage)
            
            ### YOUR CODE END HERE ###
            
            if self.num_param_updates % self.target_update_freq == 0:
                self.exploitation_critic.update_target_network()
                self.exploration_critic.update_target_network()
            
            
            # Logging #
            log['Exploration Critic Loss'] = exploration_critic_loss['Training Loss']
            log['Exploitation Critic V Loss'] = exploitation_critic_loss['Training Q Loss']
            log['Exploitation Critic Q Loss'] = exploitation_critic_loss['Training V Loss']
            log['Exploration Model Loss'] = expl_model_loss
            log['Actor Loss'] = actor_loss

            self.num_param_updates += 1

        self.t += 1
        return log


    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """
        if (not self.offline_exploitation) or (self.t <= self.num_exploration_steps):
            self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)

        perform_random_action = np.random.random() < self.eps or self.t < self.learning_starts

        if perform_random_action:
            action = self.env.action_space.sample()
        else:
            processed = self.replay_buffer.encode_recent_observation()
            action = self.actor.get_action(processed)

        next_obs, reward, done, info = self.env.step(action)
        self.last_obs = next_obs.copy()

        if (not self.offline_exploitation) or (self.t <= self.num_exploration_steps):
            self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)

        if done:
            self.last_obs = self.env.reset()
