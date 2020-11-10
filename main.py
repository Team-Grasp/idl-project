import numpy as np
import time
import datetime

import gym
import rlbench.gym

from rlbench.task_environment import TaskEnvironment, BoundaryError, WaypointError, TaskEnvironmentError
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video

# from agent import Agent
# import gym_rlbench_interface
from utils import parse_arguments
from progress_callback import ProgressCallback


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


def run_episode(model, env, max_iters):
    done = False
    obs = env.reset()
    episode_rewards = []
    i = 0 
    while not done and i < max_iters:
        # RLBench env doesn't have render
        # if render:
        #     env.render()
            
        # _states are only useful when using LSTM policies
        action, _states = model.predict(obs)
        # info isn't returned by RLBench env
        obs, reward, done = env.step(action)
        episode_rewards.append(reward)

        i+=1

    return episode_rewards


if __name__ == "__main__":
    
    args = parse_arguments()
    render = args.render
    is_train = args.train
    model_path = args.model_path
    num_episodes = args.num_episodes
    lr = args.lr
    timestamp = int(time.time())
    
    # obs_config = ObservationConfig()
    # # turn on additional cameras from robot POV, use if end-to-end
    # obs_config.set_all(False)  
    # obs_config.gripper_pose = True

    # action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
    # env = Environment(
    #     action_mode, obs_config=obs_config, headless=not render)

    # TaskEnvironment
    env = gym.make('reach_target-state-v0') #, render_mode="human")

    # agent
    print(args)
    model = PPO(MlpPolicy, env, learning_rate=lr, verbose=1, tensorboard_log="runs/")
    save_path = "models/%d" % timestamp
    # callback = ProgressCallback(eval_env=env, save_freq=500, render_freq=-1, save_path=save_path, deterministic=True, verbose=1)

    if model_path != "":
        print("Loading Existing model: %s" % model_path)
        model.load(model_path)

    if is_train:
        eval_freq = 500
        n_eval_episodes = 10
        model.learn(total_timesteps=num_episodes, eval_freq=eval_freq) #, callback=callback, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes)
        model.save("models/weights_%d" % timestamp)
    else:
        for i in range(5):
            run_episode(model, env, max_iters=100)

    # env.shutdown()


