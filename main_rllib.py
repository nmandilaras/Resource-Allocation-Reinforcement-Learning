from rdt_env import Rdt
import logging.config
from utils.argparser import cmd_parser, config_parser
import ray
from ray.rllib import agents as ray_agents
from ray import tune
from ray.tune import grid_search

parser = cmd_parser()
args = parser.parse_args()
config_env, config_agent, config_misc = config_parser(args.config_file)

logging.config.fileConfig('logging.conf')
log = logging.getLogger('simpleExample')

# trainer = ray_agents.dqn.DQNTrainer(env="CartPole-v0", config={
#     "use_pytorch": True,
#     "env_config": config_env,  # config to pass to env class
# })
# results = trainer.train()
#
config = {
    "env": Rdt,
    "use_pytorch": True,
    "env_config": config_env,
    'num_workers': 1,
}

tune.run(ray_agents.ppo.PPOTrainer, config=config)
