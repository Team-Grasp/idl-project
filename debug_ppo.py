import argparse
import torch 
import random
import numpy as np
from rlbench.action_modes import ArmActionMode
from algo.ppo_helpers import PPOWorkerHandler, PPOWorker
from algo.policy import CustomPolicy
from env.reach_task import ReachTargetCustom
from algo.ppo_helpers import PPOWorker
from env.multitask_env import MultiTaskEnv

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', dest='model_path', type=str,
                        default="", help="Existing model to use.")

parser.add_argument('--render', dest='render',
                        action='store_true',
                        help="Render env.")

args = parser.parse_args()

seed = 12345
render = args.render
model_path = args.model_path

lr = 3e-4
episode_length = 200  # horizon H
num_episodes = 5  # "K" in K-shot learning
n_steps = num_episodes * episode_length
n_epochs = 2
batch_size = 64
num_iters = 300

total_timesteps = 1 * n_steps  # number of "epochs"
action_size = 3  # only control EE position
manual_terminate = True
penalize_illegal = True

# Logistical parameters
verbose = 1
save_targets = True  # save the train targets (loaded or generated)
save_freq = 1  # save model weights every save_freq iteration

num_tasks = 10
task_batch_size = 1
act_mode = ArmActionMode.DELTA_EE_POSE_PLAN_WORLD_FRAME
alpha = 1e-3
beta = 1e-3
vf_coef = 0.5
ent_coef = 0.01

num_workers  = task_batch_size

render_mode = "human" if render else None
base_path = "./models/PPO/"

# log results
config = {
    "num_tasks": num_tasks,
    "task_batch_size": task_batch_size,
    "alpha": alpha,
    "beta": beta,
    "seed": seed,
    'n_steps': n_steps, 
    'n_epochs': n_epochs, 
    'learning_rate': alpha,
    'batch_size': batch_size, 
    'verbose': verbose, 
    'vf_coef': vf_coef, 
    'ent_coef': ent_coef,
    'total_timesteps': total_timesteps, 
    "n_steps": n_steps,
    'act_mode': act_mode, 
    "render_mode": render_mode,
    'epsiode_length': episode_length, 
    'action_size': action_size,            
    'manual_terminate': manual_terminate, 
    'penalize_illegal': penalize_illegal,
    "base_path": base_path,
    "num_workers": num_workers,
    "model_path": model_path,
    "num_iters": num_iters,
    
    'policy': CustomPolicy, 
    'task_class': ReachTargetCustom, 
    "worker_class": PPOWorker,
    "env_class": MultiTaskEnv,
}

random.seed(seed)       # python random seed
torch.manual_seed(seed)  # pytorch random seed
np.random.seed(seed)  # numpy random seed
torch.backends.cudnn.deterministic = True

import time

agent = PPOWorkerHandler(config, **config)

for i in range(10):        
    new_tasks = [None]
    agent.set_task(new_tasks=new_tasks)
    agent.evaluate(num_episodes=1, max_iters=80)
    time.sleep(1)

agent.close()


