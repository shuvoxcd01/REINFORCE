from abc import ABC, abstractmethod


class BaseEnvironment(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def get_random_action(self):
        pass

    @abstractmethod
    def get_observation_shape(self):
        pass

    @abstractmethod
    def get_num_actions(self):
        pass

