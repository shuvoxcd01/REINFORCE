import os

import tensorflow as tf

from saved_models import saved_model_dir
from src.environment.impl.cartpole_environment import CartPoleEnvironment
from src.utils.policy_evaluator import PolicyEvaluator

epoch_num = 1

eval_policy_path = os.path.join(saved_model_dir, f"_epoch_{epoch_num}.h5")
saved_video_filename = f"epoch_{epoch_num}"

eval_policy = tf.keras.models.load_model(eval_policy_path)
eval_environment = CartPoleEnvironment()
evaluator = PolicyEvaluator()
evaluator.create_policy_evaluation_video(policy=eval_policy, eval_env=eval_environment, filename=saved_video_filename)
eval_environment.close()
