from src.environment.base_environment import BaseEnvironment
import gym


class CartPoleEnvironment(BaseEnvironment):
    def __init__(self):
        self.env = gym.make("CartPole-v1")

    def reset(self):
        return self.env.reset()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        reward = reward if not done else -10

        return observation, reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def get_random_action(self):
        return self.env.action_space.sample()

    def get_observation_shape(self):
        return self.env.observation_space.shape

    def get_num_actions(self):
        return self.env.action_space.n
