import numpy as np
import time
import datetime

import torch

from rlbench.action_modes import ArmActionMode

from stable_baselines3 import PPO, HER
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from utils import parse_arguments
from progress_callback import ProgressCallback

from reach_task import ReachTargetCustomStatic, ReachTargetCustom
from grasp_env import GraspEnv

from maml import MAML


def evaluate(model, env, num_episodes=100, max_iters=500):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = run_episode(model, env, max_iters)
        total_reward = sum(episode_rewards)
        all_episode_rewards.append(total_reward)

    mean_reward = np.mean(all_episode_rewards)
    std_reward = np.std(all_episode_rewards)

    return mean_reward, std_reward


def run_episode(model, env, max_iters, render=False):
    done = False
    obs = env.reset()
    episode_rewards = []
    i = 0
    while not done and i < max_iters:
        # RLBench env doesn't have render
        if render:
            env.render()

        # _states are only useful when using LSTM policies
        action, _states = model.predict(obs)
        # info isn't returned by RLBench env
        obs, reward, done, _ = env.step(action)
        episode_rewards.append(reward)

        i += 1

    return episode_rewards


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

    episode_length = 200
    num_episodes = 5
    n_steps = num_episodes * episode_length
    total_timesteps = 1 * n_steps  # number of "epochs"
    n_epochs = 1
    batch_size = 64
    save_freq = 10
    action_size = 7  # only control EE position
    manual_terminate = True
    penalize_illegal = False

    # MAML parameters
    num_tasks = 2
    task_batch_size = 1
    act_mode = ArmActionMode.DELTA_EE_POSE_PLAN_WORLD_FRAME
    if render:
        env = GraspEnv(task_class=ReachTargetCustom, render_mode="human",
                       act_mode=act_mode, epsiode_length=episode_length, action_size=action_size,
                       manual_terminate=manual_terminate, penalize_illegal=penalize_illegal)
    else:
        env = GraspEnv(task_class=ReachTargetCustom, act_mode=act_mode,
                       epsiode_length=episode_length, action_size=action_size,
                       manual_terminate=manual_terminate, penalize_illegal=penalize_illegal)

    alpha = 1e-3
    beta = 1e-3
    vf_coef = 0.5
    ent_coef = 0.01
    verbose = 1
    num_iters = 400
    base_init_kwargs = {'policy': CustomPolicy, 'env': env, 'n_steps': n_steps, 'n_epochs': n_epochs,
                        'batch_size': batch_size, 'verbose': verbose, 'vf_coef': vf_coef, 'ent_coef': ent_coef}
    base_adapt_kwargs = {'total_timesteps': total_timesteps}

    model = MAML(BaseAlgo=PPO, num_tasks=num_tasks, task_batch_size=task_batch_size,
                 alpha=alpha, beta=beta, base_init_kwargs=base_init_kwargs, base_adapt_kwargs=base_adapt_kwargs)

    # Run one episode
    # run_episode(model, env, max_iters=100, render=True)

    # import ipdb; ipdb.set_trace()

    save_path = "models/%d" % timestamp
    callback = ProgressCallback(eval_env=env, save_freq=save_freq, render_freq=0,
                                save_path=save_path, deterministic=True, verbose=1)

    if model_path != "":
        print("Loading Existing model: %s" % model_path)
        model.model = model.model.load(model_path, env=env)

    if is_train:
        model.learn(num_iters=num_iters)
        # model.save("models/weights_%d" % timestamp)

    else:
        for i in range(5):
            run_episode(model, env, max_iters=200)

    env.close()
