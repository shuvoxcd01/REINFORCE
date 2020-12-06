from saved_models import saved_model_dir
from src.environment.impl.cartpole_environment import CartPoleEnvironment
from src.policy.reinforce import Reinforce
from src.utils.summary_writer import SummaryWriter

environment = CartPoleEnvironment()
summary_writer = SummaryWriter()
save_model_path = saved_model_dir

agent_policy = Reinforce(environment, summary_writer, save_model_path=save_model_path)

agent_policy.learn_optimal_policy()

environment.close()
