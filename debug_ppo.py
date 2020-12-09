import argparse
import random

import numpy as np
import torch
import wandb
from rlbench.action_modes import ArmActionMode

from algo.policy import CustomPolicy
from algo.ppo_helpers import PPOWorker, PPOWorkerHandler
from env.multitask_env import MultiTaskEnv
from env.reach_task import ReachTargetCustom

parser = argparse.ArgumentParser()

parser.add_argument('--algo_name', dest='algo_name', type=str,
                        required=True, help="algo_name for logging")

parser.add_argument('--train', dest='is_train',
                        action='store_true', help="Training mode On")

parser.add_argument('--model_path', dest='model_path', type=str,
                        default="", help="Existing model to use.")

parser.add_argument('--seed', dest='seed', type=int,
                        default=12345, help="Seed value")

parser.add_argument('--render', dest='render',
                        action='store_true',
                        help="Render env.")

args = parser.parse_args()

seed = args.seed
render = args.render
model_path = args.model_path
is_train = args.is_train
algo_name = args.algo_name

render_mode = "human" if render else None
base_path = "./models/" + algo_name + "/"
is_wandb = True

action_size = 3  # only control EE position
manual_terminate = True
penalize_illegal = True
act_mode = ArmActionMode.DELTA_EE_POSE_PLAN_WORLD_FRAME

verbose = 1
save_targets = True  # save the train targets (loaded or generated)
save_freq = 1  # save model weights every save_freq iteration
eval_freq = 1 

lr = 3e-4

n_epochs = 1
num_episodes = 1  # "K" in K-shot learning
batch_size = 10
num_iters = 2
episode_length = 200  # horizon H
n_steps = num_episodes * episode_length
total_timesteps = 1 * n_steps  # number of "epochs"

task_batch_size = 1
num_tasks = 2
num_workers = task_batch_size

alpha = 1e-3
beta = 1e-3
vf_coef = 0.5
ent_coef = 0.01

if is_train:
    max_tasks = len(MultiTaskEnv.train_targets)
else:
    max_tasks = len(MultiTaskEnv.test_targets)

# Checks
assert num_workers == task_batch_size

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
    "save_targets": save_targets,
    'verbose': verbose, 
    "save_freq": save_freq,
    'eval_freq': eval_freq,
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
    'is_wandb': is_wandb,
    'max_tasks': max_tasks,

    'policy': CustomPolicy, 
    'task_class': ReachTargetCustom, 
    "worker_class": PPOWorker,
    "env_class": MultiTaskEnv,
}

random.seed(seed)           # python random seed
torch.manual_seed(seed)     # pytorch random seed
np.random.seed(seed)        # numpy random seed
torch.backends.cudnn.deterministic = True

if is_wandb:
    run_title = f"idl-{algo_name}-" + "train" if is_train else "eval"
    run = wandb.init(project=run_title, entity="idl-project", config=config)
    print("Run Name: ", run.name)
    base_path = base_path + str(run.name) + "/"
    config["base_path"] = base_path
    wandb.save("algo/ppo_helpers.py")
    
import time

agent = PPOWorkerHandler(config, **config)

if is_train:
    agent.learn(config, **config)
else:
    agent.set_random_tasks()
    # agent.sample_random_tasks()
    agent.evaluate(num_episodes=1, max_iters=200)

# for i in range(10):
#     new_tasks = [i]
#     agent.set_task(new_tasks=new_tasks)
#     agent.evaluate(num_episodes=1, max_iters=200)
#     time.sleep(1)

agent.close()

