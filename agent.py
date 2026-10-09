"""DDPG agent for continuous control in Unity ML-Agents environments.

Implements Deep Deterministic Policy Gradient with:
  - actor and critic networks plus slowly-updated target networks
  - a shared experience replay buffer across parallel agents
  - Ornstein-Uhlenbeck noise for temporally correlated exploration
  - gradient-norm clipping on the critic update
"""

import numpy as np
import random
import copy
from collections import namedtuple, deque
from model import Actor, Critic
from sklearn.base import BaseEstimator
import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0.00     # L2 weight decay
TIME = BATCH_SIZE       # train the networks every TIME time steps
TRAIN_NUM = 64          # number of gradient updates per training step
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(BaseEstimator):
    """Interacts with and learns from the environment (DDPG)."""

    def __init__(self, state_size, action_size, random_seed, num_agents):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            num_agents (int): number of parallel agents sharing this learner
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.num_agents = num_agents
        print("Number of agents ", num_agents)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise((self.num_agents, action_size), random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        # Start target networks as exact copies of the local networks
        self.deep_copy(self.actor_target, self.actor_local)
        self.deep_copy(self.critic_target, self.critic_local)

    def step(self, states, actions, rewards, next_states, dones, time_stamp):
        """Save experience in replay memory, and use a random sample from the buffer to learn.

        Experience from all parallel agents is stored. Learning runs every
        TIME time steps, performing TRAIN_NUM gradient updates per step.
        Transitions with NaN or extremely negative rewards are skipped to
        keep corrupted environment feedback out of the buffer.
        """
        # Save experience / reward
        t = time_stamp % TIME

        for i in range(self.num_agents):
            if np.isnan(rewards[i]) == False and rewards[i] > -30:
                self.memory.add(states[i, :], actions[i, :], rewards[i], next_states[i, :], dones[i])

        # Learn, if enough samples are available in memory
        if t == 0 and len(self.memory) > BATCH_SIZE:
            for _ in range(TRAIN_NUM):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Return actions for the given state as per the current policy.

        Runs the actor in eval mode (so batch normalization uses its running
        statistics), then optionally adds Ornstein-Uhlenbeck exploration noise.
        Actions are clipped to [-1, 1].
        """
        state = torch.from_numpy(state).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            noise = self.noise.sample()
            if True not in np.isnan(noise.reshape(-1, 1).squeeze()):
                action += noise
            else:
                self.noise.reset()

        return np.clip(action, -1, 1)

    def reset(self):
        """Reset the exploration noise process (call at the start of each episode)."""
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using a batch of experience tuples.

        Q_targets = r + gamma * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Ask the target actor for the best believed next action ...
        actions_next = self.actor_target(next_states)
        # ... and have the target critic evaluate its Q-value
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Build the Bellman target: current reward + discounted next Q-value
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compare against what the local critic currently predicts
        Q_expected = self.critic_local(states, actions)
        # Minimize the mean-squared error between the two
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)  # keep gradients bounded
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # The actor improves by maximizing the critic's Q-value for its actions
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- soft-update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.

        theta_target = tau * theta_local + (1 - tau) * theta_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def deep_copy(self, target, source):
        """Copy all parameters from source model into target model (hard update)."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration noise."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process.

        Params
        ======
            size: shape of the noise sample, e.g. (num_agents, action_size)
            seed (int): random seed
            mu (float): long-running mean of the process
            theta (float): pull strength back toward the mean
            sigma (float): scale of the random perturbation
        """
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to the mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update the internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory, as torch tensors on `device`."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
