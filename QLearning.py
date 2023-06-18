from CarARS2 import CarARS2
import numpy as np
import matplotlib.pyplot as plt


class CarQlearning(CarARS2):
    """Simulated car"""

    def __init__(self):
        super(CarQlearning, self).__init__()
        # Q-learning
        self.state_value_bounds = [np.pi / 8 * np.array([-1, 1]), (-3, 3)] # The first state is the line angle, with a range of [-pi/8, pi/8], the second is the u vanishing coordinate
        # And no second state considered since the second tuple is (0,0)
        self.no_buckets = (4, 1) # we divide the state into 2 intervals, thus they are : [-pi/8, 0] & (0, pi/8]
        self.no_actions = 4 # 4 actions : turn left 1 deg & turn right 1 deg & turn left 2 deg & turn right 2 deg
        self.q_value_table = np.zeros(self.no_buckets + (self.no_actions,))

        # user-defined parameters
        self.min_explore_rate = 0.002
        self.min_learning_rate = 0.012
        self.max_episodes = 5000
        self.streak_to_end = 300
        self.discount = 0.99
        self.no_streaks = 0

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

    def bucketize_state_value(self, state_value):
        """ Discretizes continuous values into fixed buckets"""
        bucket_indices = []
        for i in range(len(state_value)):

            if state_value[i] <= self.state_value_bounds[i][0]:  # violates lower bound
                bucket_index = 0

            elif state_value[i] >= self.state_value_bounds[i][1]:  # violates upper bound
                bucket_index = self.no_buckets[i] - 1  # put in the last bucket

            else:
                bound_width = self.state_value_bounds[i][1] - self.state_value_bounds[i][0]
                offset = (self.no_buckets[i] - 1) * self.state_value_bounds[i][0] / bound_width
                scaling = (self.no_buckets[i] - 1) / bound_width
                bucket_index = int(round(scaling * state_value[i] - offset))

            bucket_indices.append(bucket_index)

        return tuple(bucket_indices)

    def get_state_value(self):
        theta, dist = self.hough_param_bisect()
        u_vanish, v_vanish = self.vanish_coord()

        return self.bucketize_state_value((theta, u_vanish))

    def select_action(self, state_value, explore_rate):
        if np.random.random() < explore_rate:
            action = np.random.randint(self.no_actions)  # explore
        else:
            action = np.argmax(self.q_value_table[state_value])  # exploit

        return action

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
        # change the exploration rate over time.
        return max(self.min_explore_rate, min(1.0, 1.0 - np.log10((x + 1) / 25)))

    def select_learning_rate(self, x):
        # Change learning rate over time
        return max(self.min_learning_rate, min(1.0, 1.0 - np.log10((x + 1) / 50)))


if __name__ == "__main__":

    car = CarQlearning()

    frames = []
    reward_per_episode = []
    dist_per_episode = []
    avgdist_per_episode = []
    learning_rate_per_episode = []
    explore_rate_per_episode = []

    # train the system
    totaldist = 0
    for episode_no in range(car.max_episodes):

        explore_rate = car.select_explore_rate(episode_no)
        learning_rate = car.select_learning_rate(episode_no)

        learning_rate_per_episode.append(learning_rate)
        explore_rate_per_episode.append(explore_rate)

        # reset the environment while starting a new episode
        car.reset_state()

        start_state_value = car.get_state_value()
        previous_state_value = start_state_value

        while car.valid_state():
            # env.render()
            action = car.select_action(previous_state_value, explore_rate)
            car.make_action(action)
            car.update_state()
            reward_gain = car.valid_state()

            state_value = car.get_state_value()
            best_q_value = np.max(car.q_value_table[state_value])

            # update q_value_table
            car.q_value_table[previous_state_value][action] += learning_rate * (
                reward_gain
                + car.discount * best_q_value
                - car.q_value_table[previous_state_value][action]
            )

            previous_state_value = state_value
            # while loop ends here

        if car.x >= car.x_max:
            car.no_streaks += 1
        else:
            car.no_streaks = 0

        if car.no_streaks > car.streak_to_end:
            print("Lane keeping problem is solved after {} episodes.".format(episode_no))
            break

        # data log
        if episode_no % 100 == 0:
            print(f"Episode {episode_no} finished after {np.floor(car.x)} meters")
        dist_per_episode.append(car.x)
        totaldist += car.x
        avgdist_per_episode.append(totaldist / (episode_no + 1))
        # episode loop ends here

    # Plotting
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(dist_per_episode)
    axes[0].plot(avgdist_per_episode)
    axes[0].set(ylabel="distance per episode")
    axes[1].plot(learning_rate_per_episode, label='Learning rate')
    axes[1].plot(explore_rate_per_episode, label='Exploration rate')
    axes[1].set_ylim([0, 1])
    axes[1].set(xlabel="Episodes", ylabel="Learning rate")
    axes[1].legend()
    plt.show()

    # Show final result
    car.reset_state()
    car.render = True

    car.display_init()
    line = None
    while car.valid_state():
        car.display_update()
        try:
            line.remove()
        except:
            pass

        # Show Hough line bisector
        angle, dist = car.hough_param_bisect()
        line = car.cam.axline(
            car.vanish_coord(), slope=np.tan(angle + np.pi / 2), color="red",
        )
        # Pause
        if not car.is_pause:
            state_value = car.get_state_value()
            action = car.select_action(state_value, 0)
            car.make_action(action)
            car.update_state()
        plt.pause(1 / car.freq)

    plt.show()
