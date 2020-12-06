import os

import imageio as imageio
import tensorflow as tf

from saved_videos import saved_video_dir
from src.environment.base_environment import BaseEnvironment
from src.policy.reinforce import Reinforce


class PolicyEvaluator:
    @staticmethod
    def create_policy_evaluation_video(policy: Reinforce, eval_env: BaseEnvironment, filename, num_episodes=5,
                                       fps=30):
        filename = filename + ".mp4"
        file_path = os.path.join(saved_video_dir, filename)

        with imageio.get_writer(file_path, fps=fps) as video:
            for _ in range(num_episodes):
                observation = eval_env.reset()
                done = False
                video.append_data(eval_env.render(mode='rgb_array'))
                while not done:
                    observation = tf.expand_dims(observation, 0)
                    action = policy.get_action(observation)
                    observation, reward, done, info = eval_env.step(action)
                    video.append_data(eval_env.render(mode='rgb_array'))
