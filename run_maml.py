import numpy as np
import time
import datetime
import copy
import sys

import torch

from rlbench.action_modes import ArmActionMode

from stable_baselines3 import PPO, HER
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from utils import parse_arguments
from progress_callback import ProgressCallback

from reach_task import ReachTargetCustom
from grasp_env import GraspEnv
from multitask_env import MultiTaskEnv

from maml import MAML

from eval_utils import *

"""Commands:
Train:
python run_maml.py --train 

Eval:
python run_maml.py --eval --model_path=models/1607153777/110_iters.zip \
--train_targets_path=models/1607153777/targets.npy --test_targets_path=test_targets.npy


"""


class CustomPolicy(MlpPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[64, 64, dict(pi=[64, 64], vf=[64, 64])])

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

    def _get_torch_save_params(self):
        state_dicts = ["policy", "policy.optimizer", "policy.lr_scheduler"]

        return state_dicts, []


if __name__ == "__main__":

    sys.stdout = open("outputs.txt", "w")

    # Args
    args = parse_arguments()
    render = args.render
    is_train = args.train
    model_path = args.model_path
    train_targets_path = args.train_targets_path
    test_targets_path = args.test_targets_path
    num_episodes = args.num_episodes
    lr = args.lr
    # lr_scheduler = None
    timestamp = int(time.time())
    print(args)

    # MAML Iterations:
    # for iter = 1:num_iters
    #     for task in task_batch (size=task_batch_size)
    #         During PPO's Adaption to a specific task:
    #         for t = 1:total_timesteps
    #             for e = 1:num_episodes:
    #                 Gen + add new episode of max episode_length
    #             for epoch = 1:n_epochs
    #                 for batch (size=batch_size) in batches
    #                     Calc loss over collected episodes, step gradients
    #             Post-update collection of new data and gradients:
    #             for e = 1:num_episodes:
    #                 Gen + add new episode of max episode_length
    #             Gradients += this task's PPO Gradients
    #     Step with summed gradients

    # PPO Adaptation Parameters
    episode_length = 200  # horizon H
    num_episodes = 5  # "K" in K-shot learning
    n_steps = num_episodes * episode_length
    total_timesteps = 1 * n_steps  # number of "epochs"
    n_epochs = 1
    batch_size = None
    action_size = 3  # only control EE position
    manual_terminate = True
    penalize_illegal = True

    # Logistical parameters
    verbose = 1
    save_targets = True  # save the train targets (loaded or generated)
    save_freq = 10  # save model weights every save_freq iteration

    # MAML parameters
    num_iters = 400
    num_tasks = 10
    task_batch_size = 8
    act_mode = ArmActionMode.DELTA_EE_POSE_PLAN_WORLD_FRAME
    alpha = 1e-3
    beta = 1e-3
    vf_coef = 0.5
    ent_coef = 0.01
    base_init_kwargs = {'policy': CustomPolicy, 'n_steps': n_steps, 'n_epochs': n_epochs, 'learning_rate': alpha,
                        'batch_size': batch_size, 'verbose': verbose, 'vf_coef': vf_coef, 'ent_coef': ent_coef}
    base_adapt_kwargs = {'total_timesteps': total_timesteps}
    render_mode = "human" if render else None
    env_kwargs = {'task_class': ReachTargetCustom, 'act_mode': act_mode, "render_mode": render_mode,
                  'epsiode_length': episode_length, 'action_size': action_size,
                  'manual_terminate': manual_terminate, 'penalize_illegal': penalize_illegal}
    save_path = "models/%d" % timestamp
    save_kwargs = {'save_freq': save_freq,
                   'save_path': save_path, 'tensorboard_log': save_path, 'save_targets': save_targets}

    # load in targets
    try:
        train_targets = np.load(train_targets_path)
        task_batch_size = min(task_batch_size, len(train_targets))
        print("Loaded train targets from: %s" % train_targets_path)
    except FileNotFoundError as e:
        print("Failed to load train_targets, auto-generating new ones due to %s" % e)
        train_targets = None
    try:
        test_targets = np.load(test_targets_path)
        print("Loaded test targets from: %s" % test_targets_path)
    except FileNotFoundError as e:
        print("Failed to load test_targets, auto-generating new ones due to %s" % e)
        test_targets = None

    # create maml class that spawns multiple agents and sim environments
    model = MAML(BaseAlgo=PPO, EnvClass=GraspEnv,
                 num_tasks=num_tasks, task_batch_size=task_batch_size, targets=train_targets,
                 alpha=alpha, beta=beta, model_path=model_path,
                 env_kwargs=env_kwargs, base_init_kwargs=base_init_kwargs, base_adapt_kwargs=base_adapt_kwargs)

    if is_train:
        model.learn(num_iters=num_iters, save_kwargs=save_kwargs)

    else:
        assert(test_targets is not None)

        # see performance on train tasks
        pre_metrics, post_metrics = model.eval_performance(targets=None)
        print("Train Results:")
        print("pre_metrics", pre_metrics)
        print("post_metrics", post_metrics)

        # see performance on test tasks
        pre_metrics, post_metrics = model.eval_performance(
            targets=test_targets)
        print("Test Results:")
        print("pre_metrics", pre_metrics)
        print("post_metrics", post_metrics)

    model.close()
