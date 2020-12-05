from src.environment.impl.cartpole_environment import CartPoleEnvironment
from src.policy.reinforce import Reinforce
from src.utils.summary_writer import SummaryWriter

environment = CartPoleEnvironment()
summary_writer = SummaryWriter()

agent_policy = Reinforce(environment, summary_writer)

agent_policy.learn_optimal_policy()

environment.close()
