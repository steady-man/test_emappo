import gym
from gym import spaces
import numpy as np
from envs.env_core import EnvCore


class DiscreteActionEnv(object):

    def __init__(self):
        self.env = EnvCore()
        self.num_agent = self.env.agent_num

        self.signal_obs_dim = self.env.obs_dim

        self.signal_buffer_obs_dim = self.env.buffer_obs_dim

        self.signal_action_dim = self.env.action_dim

        self.discrete_action_input = False

        self.movable = True

        self.action_space = []
        self.observation_space = []
        self.buffer_observation_space = []
        self.share_observation_space = []

        share_obs_dim = 0
        total_action_space = []
        for agent in range(self.num_agent):

            u_action_space = spaces.Discrete(self.signal_action_dim)

            if self.movable:
                total_action_space.append(u_action_space)

            if len(total_action_space) > 1:

                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])

            share_obs_dim += self.signal_buffer_obs_dim

            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(self.signal_obs_dim,),
                                                     dtype=np.float16))
            self.buffer_observation_space.append(
                spaces.Box(low=-np.inf, high=+np.inf, shape=(self.signal_buffer_obs_dim,),
                           dtype=np.float16))

        self.share_observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,),
                                                   dtype=np.float16) for _ in range(self.num_agent)]

    def step(self, actions):
        results = self.env.step(actions)
        obs, rews, dones, infos = results
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        obs = self.env.reset()
        return np.stack(obs)

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        pass

    def seed(self, seed):
        pass


class MultiDiscrete():

    def __init__(self, array_of_param_array):
        super().__init__()
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        self.num_discrete_space = self.low.shape[0]
        self.n = np.sum(self.high) + 2

    def sample(self):
        random_array = np.random.rand(self.num_discrete_space)
        return [int(x) for x in np.floor(np.multiply((self.high - self.low + 1.), random_array) + self.low)]

    def contains(self, x):
        return len(x) == self.num_discrete_space and (np.array(x) >= self.low).all() and (
                np.array(x) <= self.high).all()

    @property
    def shape(self):
        return self.num_discrete_space

    def __repr__(self):
        return "MultiDiscrete" + str(self.num_discrete_space)

    def __eq__(self, other):
        return np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)


if __name__ == "__main__":
    DiscreteActionEnv().step(actions=None)
