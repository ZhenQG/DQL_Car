from CarARS2 import CarARS2
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class CarDQlearning(CarARS2):
    """Simulated car"""

    def __init__(self):
        super(CarDQlearning, self).__init__()
        # Params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Params
        self.streak_to_end = 100
        self.no_streaks = 0
        # Deep Q-learning
        self.n_observations = 1 # Dimension of state, 1 if only one state (yaw angle), 2 if considering the u_vanishing 
        self.n_actions = 4

        self.BATCH_SIZE = 128
        self.GAMMA = 0.99

        # Exploratino rate
        self.EPS_START = 0.05
        self.EPS_END = 0.001
        self.EPS_DECAY = 1000

        self.TAU = 0.005
        self.LR = 1e-4

        # Deep Networks
        self.policy_net = self.DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net = self.DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Optimizer
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = self.ReplayMemory(10000)

        # Animation parameters
        self.render = False

    def hough_param_bisect(self):
        # Simulated hough lines parameters of the center of the lane
        du = self.cam_fu * self.y / np.cos(self.theta)
        dv = self.cam_fv
        u0, v0 = self.vanish_coord()
        theta = -np.arctan2(du, dv)
        dist = (u0 * dv - v0 * du) / np.sqrt(du ** 2 + dv ** 2)

        return theta, dist

    def get_state_value(self):
        theta, dist = self.hough_param_bisect()
        u_vanish, v_vanish = self.vanish_coord()

        return np.array([theta], dtype=np.float32)

    def select_action(self, state_value, explore_rate):
        if np.random.random() < explore_rate:
            action = torch.tensor(
                [[np.random.randint(self.n_actions)]],
                device=self.device,
                dtype=torch.long,
            )  # explore
        else:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action = self.policy_net(state_value).max(1)[1].view(1, 1)

        return action

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        
        transitions = self.memory.sample(self.BATCH_SIZE)
        Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def make_action(self, action):
        if action == 0:
            self.turn_left()
        elif action == 1:
            self.turn_right()
        elif action == 2:
            self.turn_left(dtheta=2*np.pi/180)
        elif action == 3:
            self.turn_right(dtheta=2*np.pi/180)

    def select_explore_rate(self, x):
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * np.exp(-1.0 * x / self.EPS_DECAY)
        # change the exploration rate over time.

        return eps_threshold

    class ReplayMemory(object):
        def __init__(self, capacity):
            self.memory = deque([], maxlen=capacity)

        def push(self, *args):
            """Save a transition"""
            Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
            self.memory.append(Transition(*args))

        def sample(self, batch_size):
            return random.sample(self.memory, batch_size)

        def __len__(self):
            return len(self.memory)

    class DQN(nn.Module):
        def __init__(self, n_observations, n_actions):
            super(CarDQlearning.DQN, self).__init__()
            self.layer1 = nn.Linear(n_observations, 16)
            self.layer2 = nn.Linear(16, 32)
            self.layer3 = nn.Linear(32, 16)
            self.layer4 = nn.Linear(16, n_actions)

        # Called with either one element to determine next action, or a batch
        # during optimization. Returns tensor([[left0exp,right0exp]...]).
        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            x = F.relu(self.layer3(x))

            return self.layer4(x)


if __name__ == "__main__":

    car = CarDQlearning()

    if torch.cuda.is_available():
        num_episodes = 2000
    else:
        num_episodes = 1000
        
        
    dist_per_episode = []
    avgdist_per_episode = []
    explore_rate_per_episode = []
    
    totaldist = 0
    for episode_no in range(num_episodes):
        # Initialize the environment and get it's state
        car.reset_state()
        state = torch.tensor(car.get_state_value(), dtype=torch.float32, device=car.device).unsqueeze(0)
                
        explore_rate = car.select_explore_rate(episode_no)   
        explore_rate_per_episode.append(explore_rate)  
        
        while car.valid_state():
            action = car.select_action(state, explore_rate)
            car.make_action(action.item())
            car.update_state()
            reward_gain = 1 if car.valid_state() else 0

            observation = car.get_state_value()
            reward = torch.tensor([reward_gain], device=car.device)

            if not car.valid_state():
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=car.device).unsqueeze(0)
                
            # Store the transition in memory
            car.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            car.optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = car.target_net.state_dict()
            policy_net_state_dict = car.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * car.TAU + target_net_state_dict[key] * (1 - car.TAU)
            car.target_net.load_state_dict(target_net_state_dict)

        if car.x >= car.x_max:
            car.no_streaks += 1
        else:
            car.no_streaks = 0

        if car.no_streaks > car.streak_to_end:
            print("Problem is solved after {} episodes.".format(episode_no))
            break

        # data log
        if episode_no % 100 == 0:
            print(f"Episode {episode_no} finished after {np.floor(car.x)} meters")
        dist_per_episode.append(car.x)
        totaldist += car.x
        avgdist_per_episode.append(totaldist / (episode_no + 1))

    # Plotting
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(dist_per_episode)
    axes[0].plot(avgdist_per_episode)
    axes[0].set(ylabel="Distance per episode")
    axes[1].plot(explore_rate_per_episode)
    axes[1].set_ylim([0, 1])
    axes[1].set(xlabel="Episodes", ylabel="Exploration rate")
    plt.show()

    # Show final result
    car.reset_state()
    car.render = True

    car.display_init()
    while car.valid_state():
        car.display_update()
        # Pause
        if not car.is_pause:
            state_value = torch.tensor(car.get_state_value(), dtype=torch.float32, device=car.device).unsqueeze(0)
            action = car.select_action(state_value, 0)
            car.make_action(action.item())
            car.update_state()
        plt.pause(1 / car.freq)

    plt.show()
