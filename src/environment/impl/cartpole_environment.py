import gym

from src.environment.base_environment import BaseEnvironment


class CartPoleEnvironment(BaseEnvironment):
    def __init__(self):
        self.env = gym.make("CartPole-v0")
        self.max_num_step = 20
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return self.env.reset()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        self.step_count += 1

        if self.step_count >= self.max_num_step:
            done = True

        reward = reward if not done else -1

        return observation, reward, done, info

    def render(self, *args, **kwargs):
        self.env.render(*args, **kwargs)

    def close(self):
        self.env.close()

    def get_random_action(self):
        return self.env.action_space.sample()

    def get_observation_shape(self):
        return self.env.observation_space.shape

    def get_num_actions(self):
        return self.env.action_space.n
