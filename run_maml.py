import numpy as np
import time
import datetime
import copy

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

    # Args
    args = parse_arguments()
    render = args.render
    is_train = args.train
    model_path = args.model_path
    num_episodes = args.num_episodes
    lr = args.lr
    # lr_scheduler = None
    timestamp = int(time.time())
    print(args)

    episode_length = 200  # horizon H
    num_episodes = 5  # "K" in K-shot learning
    n_steps = num_episodes * episode_length
    total_timesteps = 1 * n_steps  # number of "epochs"
    n_epochs = 1
    batch_size = None
    action_size = 3  # only control EE position
    manual_terminate = True
    penalize_illegal = True

    # MAML parameters
    num_tasks = 10
    task_batch_size = 3
    act_mode = ArmActionMode.DELTA_EE_POSE_PLAN_WORLD_FRAME
    # if render:
    #     env = GraspEnv(task_class=ReachTargetCustom, render_mode="human",
    #                    act_mode=act_mode, epsiode_length=episode_length, action_size=action_size,
    #                    manual_terminate=manual_terminate, penalize_illegal=penalize_illegal)
    # else:
    #     env = GraspEnv()

    render_mode = "human" if render else None
    env_kwargs = {'task_class': ReachTargetCustom, 'act_mode': act_mode, "render_mode": render_mode,
                  'epsiode_length': episode_length, 'action_size': action_size,
                  'manual_terminate': manual_terminate, 'penalize_illegal': penalize_illegal}
    alpha = 1e-3
    beta = 1e-3
    vf_coef = 0.5
    ent_coef = 0.01
    verbose = 1
    num_iters = 400
    base_init_kwargs = {'policy': CustomPolicy, 'n_steps': n_steps, 'n_epochs': n_epochs, 'learning_rate': alpha,
                        'batch_size': batch_size, 'verbose': verbose, 'vf_coef': vf_coef, 'ent_coef': ent_coef}
    base_adapt_kwargs = {'total_timesteps': total_timesteps}

    train_targets_path = "models/1607113136/targets.npy"
    train_targets = np.load(train_targets_path)
    model = MAML(BaseAlgo=PPO, EnvClass=GraspEnv, num_tasks=num_tasks, task_batch_size=task_batch_size, env_kwargs=env_kwargs, targets=train_targets,
                 alpha=alpha, beta=beta, base_init_kwargs=base_init_kwargs, base_adapt_kwargs=base_adapt_kwargs)

    # Run one episode
    # run_episode(model, env, max_iters=100, render=True)

    # import ipdb; ipdb.set_trace()
    save_targets = True
    save_freq = 10
    save_path = "models/%d" % timestamp
    save_kwargs = {'save_freq': save_freq,
                   'save_path': save_path, 'tensorboard_log': save_path, 'save_targets': save_targets}
    # callback = ProgressCallback(eval_env=env, save_freq=save_freq, render_freq=0,
    #                             save_path=save_path, deterministic=True, verbose=1)

    if model_path != "":
        print("Loading Existing model: %s" % model_path)
        # model.model = model.model.load(model_path, env=env)

    if is_train:
        model.learn(num_iters=num_iters, save_kwargs=save_kwargs)
        # model.save("models/weights_%d" % timestamp)

    else:
        # np.save("test_targets", model.targets)
        # exit()

        # load in tasks
        train_targets_path = "models/1607095661/targets.npy"
        test_targets_path = "test_targets.npy"
        train_targets = np.load(train_targets_path)
        test_targets = np.load(test_targets_path)

        # store metrics
        pre_adapt_rewards = 0.0
        post_adapt_rewards = 0.0

        # make model revert to original loaded weights after each trial of updates
        restore_weights = True

        # see how the model does using the default loaded weights
        for i in range(0, len(train_targets)):
            pre_rewards, post_rewards = model.eval_performance(
                target_position=train_targets[i], restore_weights=restore_weights)

            pre_adapt_rewards += sum(pre_rewards)
            post_adapt_rewards += sum(post_rewards)

        print("Mean Pre-adapt Total Reward: %.3f" %
              (pre_adapt_rewards / len(train_targets)))

        print("Mean Post-adapt Total Reward: %.3f" %
              (post_adapt_rewards / len(train_targets)))

        # see how model does after one round of updates and compare, this should verify if MAML is working

        # randomly generate a test task or load one. Reset weights to the original loaded ones
        # perform the same comparison of first two steps above

        # for i in range(5):
        #     run_episode(model, env, max_iters=200)

    env.close()
