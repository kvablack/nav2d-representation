import torch
import numpy as np
import torch.nn.functional as F
from collections import deque
from nav2d_representation import nets
import random
import tqdm
from copy import deepcopy


class DQN_HER:
    def __init__(
        self,
        env,
        encoder,
        dqn=None,
        gamma=0.99,
        use_ddqn=True,
        tau=0.005,
        learning_rate=0.001,
        batch_size=16,
        epsilon_high=0.9,
        epsilon_low=0.2,
        epsilon_decay=10000,
        updates_per_step=1,
        replay_buffer_size=1000000,
    ):
        self.env = env

        if dqn is None:
            dqn = nets.DQN(encoder.output_dim, env.action_space.n)

        self.model = torch.nn.Sequential(
            encoder,
            # nets.SideTuner(encoder, env.observation_space.shape),
            dqn,
        ).cuda()
        self.target_model = deepcopy(self.model).cuda()
        self.gamma = gamma
        self.ddqn = use_ddqn
        self.tau = tau
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()),
            lr=learning_rate,
        )
        self.batch_size = batch_size
        self.epsilon_high = epsilon_high
        self.epsilon_low = epsilon_low
        self.epsilon_decay = epsilon_decay
        self.updates_per_step = updates_per_step

        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.steps = 0
        self.epsilon = self.epsilon_high

    def run_episode(self):
        obs = self.env.reset()
        done = False
        total_reward = 0
        total_loss = 0
        t = 0
        min_dist = self.env.dist

        while not done:
            self.steps += 1
            t += 1
            self.epsilon = self.epsilon_low + (self.epsilon_high - self.epsilon_low) * (
                np.exp(-1.0 * self.steps / self.epsilon_decay)
            )
            if np.random.rand() < self.epsilon:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    Q = self.model(
                        torch.tensor(obs, dtype=torch.float, device="cuda").unsqueeze(0)
                    ).squeeze(0)
                    action = torch.argmax(Q).item()
            obs_next, reward, done, info = self.env.step(action)
            total_reward += reward
            min_dist = min(min_dist, self.env.dist)

            self.replay_buffer.append([obs, action, reward, obs_next])
            for _ in range(self.updates_per_step):
                loss = self.update_model()
                self.sync_params()
                total_loss += loss
            obs = obs_next

        self.replay_buffer.extend(self.env.her())
        avg_loss = total_loss / t / self.updates_per_step
        return total_reward, avg_loss, min_dist, t

    def update_model(self):
        self.optimizer.zero_grad()
        K = min(len(self.replay_buffer), self.batch_size)
        samples = random.sample(self.replay_buffer, K)

        states, actions, rewards, new_states = zip(*samples)
        states = torch.tensor(states, dtype=torch.float, device="cuda")
        actions = torch.tensor(actions, dtype=torch.long, device="cuda")
        rewards = torch.tensor(rewards, dtype=torch.float, device="cuda")
        new_states = torch.tensor(new_states, dtype=torch.float, device="cuda")

        if self.ddqn:
            model_next_acts = self.model(new_states).detach().max(dim=1)[1]
            target_q = rewards + self.gamma * self.target_model(new_states).gather(
                1, model_next_acts.unsqueeze(1)
            ).squeeze() * (rewards == -1)
        else:
            target_q = rewards + self.gamma * self.target_model(new_states).max(dim=1)[
                0
            ].detach() * (rewards == -1)
        policy_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        L = F.smooth_l1_loss(policy_q, target_q)
        L.backward()
        self.optimizer.step()
        return L.detach().item()

    def sync_params(self):
        for curr, targ in zip(self.model.parameters(), self.target_model.parameters()):
            targ.data.copy_(targ.data * (1.0 - self.tau) + curr.data * self.tau)
