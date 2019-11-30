import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

# import out models
from model import Actor, Critic
from ounoise import OUNoise
from replaybuffer import ReplayBuffer

class Agent():
    """ DDPG Agent, interacts with environment and learns from environment """
    def __init__(self, device, state_size, n_agents, action_size, random_seed, \
                         buffer_size, batch_size, gamma, TAU, lr_actor, lr_critic, weight_decay,  \
                         learn_interval, learn_num, ou_sigma, ou_theta, checkpoint_folder = './'):
        
        # Set Computational device
        self.DEVICE = device
        
        # Init State, action and agent dimensions
        self.state_size = state_size
        self.n_agents = n_agents
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.l_step = 0
        # self.log_interval = 200

        # Init Hyperparameters
        self.BUFFER_SIZE = buffer_size
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.TAU = TAU
        self.LR_ACTOR = lr_actor
        self.LR_CRITIC = lr_critic
        self.WEIGHT_DECAY = weight_decay
        self.LEARN_INTERVAL = learn_interval
        self.LEARN_NUM = learn_num
    
        # Init Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed, fc1_units=256, \
                                 fc2_units=128).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed,fc1_units=256, \
                                  fc2_units=128).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Init Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed, fcs1_units=256, \
                                   fc2_units=128).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed, fcs1_units=256, \
                                    fc2_units=128).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), \
                                    lr=lr_critic, weight_decay=weight_decay)
    
        # Init Noise Process
        self.noise = OUNoise(action_size, random_seed, mu=0., theta=ou_theta, sigma=ou_sigma)
        
        # Init Replay Memory
        self.memory = ReplayBuffer(device, action_size, buffer_size, batch_size, random_seed)
    
    # think
    def think(self, states, add_noise=True):
        """ Decide what action to take next, for both agents """
        
        # evaluate state through actor_local
        s = torch.from_numpy(states).float().to(self.DEVICE)
        
        self.actor_local.eval() # put actor_local network in "evaluation" mode
        with torch.no_grad():
            a = self.actor_local(s).cpu().data.numpy()
        self.actor_local.train() # put actor_local back into "training" mode
        
        # add noise for better performance
        if add_noise:
            a += self.noise.sample()
    
        return np.clip(a, -1, 1) # clip action output between [-1,1]
    
    # accept
    def accept(self, t, s, a, r, s_, done, agent_id):
        """ Commit step into the brain
        Params:
            t = timestep
            s = state
            a = action
            r = reward
            s_ = next state
            done = if we're done or not
        """
        
        # Save SARS' to replay buffer --- state-action-reward-next_state tuple
        self.memory.add(s, a, r, s_, done)

        if t % self.LEARN_INTERVAL != 0:
            return
            
        # Learn (if enough samples are available in memory)
        if len(self.memory) > self.BATCH_SIZE:
            for _ in range(self.LEARN_NUM):
                experiences = self.memory.sample() # get a memory sample
                self.learn(experiences, self.GAMMA, agent_id)
    
    
    def reset(self):
        self.noise.reset()
        
    def learn(self, experiences, gamma, agent_id):
        """ Learn from experiences, with discount factor gamma
        
        Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        
        Params:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        
        s, a, r, s_, dones = experiences
        
        # ------ Update Critic ------ #
        
        # get predicted next-state actions and Q values from target networks
        a_ = self.actor_target(s_)
        
        # here we'll re-order the action-space depending on the perspective of the agent
        if agent_id == 0:
            a__ = (a_, a[:, self.action_size:])
        elif agent_id == 1:
            a__ = (a[:, :self.action_size], a_)
        
        a_ = torch.cat(a__, dim=1)
            
        Q_targets_next = self.critic_target(s_, a_)
        
        # compute Q targets for current states (y_i)
        Q_targets = r + (gamma * Q_targets_next * (1 - dones))
        
        # compute critic loss
        Q_expected = self.critic_local(s, a)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # minimize loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        
        # ------ Update Actor ------ #
        
        # get next action (as predicted by actor local)
        a_ = self.actor_local(s)
        
        # here we'll re-order the predicted action-space based on the perspective of the agent
        if agent_id == 0:
            a__ = (a_, a[:, self.action_size:])
        elif agent_id == 1:
            a__ = (a[:, :self.action_size], a_)
        a_ = torch.cat(a__, dim=1)
        
        actor_loss = -self.critic_local(s, a_).mean()
        
        # minimize loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ------ Update Target Networks ------ #
        self.soft_update(self.critic_local, self.critic_target, self.TAU)
        self.soft_update(self.actor_local, self.actor_target, self.TAU)
        
        # keep count of steps taken
        # self.noise.reset()
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)