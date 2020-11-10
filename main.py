import numpy as np
import time
import datetime

from rlbench.action_modes import ArmActionMode
from rlbench.tasks import ReachTarget

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from utils import parse_arguments
from progress_callback import ProgressCallback

from grasp_env import GraspEnv

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

        i+=1

    return episode_rewards


if __name__ == "__main__":
    
    # Args
    args = parse_arguments()
    render = args.render
    is_train = args.train
    model_path = args.model_path
    num_episodes = args.num_episodes
    lr = args.lr
    timestamp = int(time.time())
    print(args)
    
    eval_freq = 300
    n_steps = 400
    total_timesteps = 200 * n_steps
    n_epochs = 20
    batch_size = 64
    save_freq = 10

    # TaskEnvironment
    # env = gym.make('reach_target-state-v0', render_mode="human")
    if render:
        env = GraspEnv(task_class=ReachTarget, render_mode="human")
    else:
        env = GraspEnv(task_class=ReachTarget, render_mode="rgb_array")

    # agent
    model = PPO(MlpPolicy, env, n_steps=n_steps, n_epochs=n_epochs, batch_size=batch_size, \
        learning_rate=lr, verbose=1, tensorboard_log="runs/")
    
    # Run one episode
    # run_episode(model, env, max_iters=10000, render=True)

    # import ipdb; ipdb.set_trace()

    save_path = "models/%d" % timestamp
    callback = ProgressCallback(eval_env=env, save_freq=save_freq, render_freq=1, \
            save_path=save_path, deterministic=True, verbose=1)

    if model_path != "":
        print("Loading Existing model: %s" % model_path)
        model.load(model_path)

    if is_train:
        model.learn(total_timesteps=total_timesteps, eval_freq=eval_freq, eval_log_path="./evals/", \
            callback=callback
            ) 
        model.save("models/weights_%d" % timestamp)

    else:

        for i in range(2):
            run_episode(model, env, max_iters=200)

    env.close()


