import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

############ ENVIRONMENT ######################

class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)
    

############# AGENTS ##########################

class RandomAgent:
    def __init__(self, env):
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        
    def compute_action(self, state):
        return np.random.uniform(-1, 1, size=(self.action_size, ))
    

class HeuristicPendulumAgent:
    def __init__(self, env, t):
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.t = t
    def compute_action(self, state):
        if state[0] <= 0:
            return np.array([self.t * np.sign(state[2])])
        else:
            return np.array([-self.t * np.sign(state[2])])
        
        
class DDPGAgent:
    def __init__(self, sigma, device, tau = 0.5, theta = 0.5, OUNoise = False):
        if OUNoise:
            self.noise = OUActionNoise(theta, sigma)
        else:
            self.noise = GaussianActionNoise(sigma)
        
        self.tau = tau
        self.device = device
        self.Q_network = QNetwork().to(self.device)
        self.policy_network = PolicyNetwork().to(self.device)
        self.target_Q_network = QNetwork().to(self.device)
        self.target_policy_network = PolicyNetwork().to(self.device)
        
    def compute_action(self, state, deterministic = True, use_target = False):
        state = torch.tensor(state, dtype=torch.float32).to(self.device) 
        if use_target:
            action = self.target_policy_network(state).detach().numpy()
        else:
            action = self.policy_network(state).detach().numpy()
        
        if not deterministic:
            action = self.noise.get_noisy_action(action)
        
        return action
    
    def update_target_params():
        for target_param, param in zip(self.target_Q_network.parameters(), self.Q_network.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        
        for target_param, param in zip(self.target_policy_network.parameters(), self.policy_network.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)        


####################### BUFFER #####################################           
            
class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        
    def store_transition(self, trans):
        if len(self.buffer)<self.max_size:           
            self.buffer.append(trans)
    
    def batch_buffer(self, batch_size):
        n = len(self.buffer)
        indexes = np.random.permutation(n)[:min(batch_size,self.max_size)]
        return [self.buffer[i] for i in indexes]
    

#################### NETWORKS ####################################
    
class QNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 32)
        self.layer2 = nn.Linear(32, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.output(x)
        return x
    
class PolicyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3, 32)
        self.layer2 = nn.Linear(32, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.output(x)
        x = torch.tanh(x)
        return x

    
######################### NOISE ########################################

class GaussianActionNoise:
    def __init__(self, sigma):
        self.sigma = sigma
        
    def get_noisy_action(self, action):
        noisy_action = action + np.random.normal(0, self.sigma)
        return np.clip(noisy_action, -1, 1)

class OUActionNoise:
    def __init__(self, theta, sigma):
        self.theta = theta
        self.sigma = sigma
        self.noise = 0
        
    def get_noisy_action(self, action):
        self.evolve_state()
        noisy_action = action + self.noise        
        return np.clip(noisy_action, -1, 1)
    
    def evolve_state():
        self.noise = (1-self.theta)*self.noise + np.random.normal(0, self.sigma)