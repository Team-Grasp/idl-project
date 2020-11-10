import numpy as np
import time

from gym import spaces

from rlbench.task_environment import TaskEnvironment
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo.policies import MlpPolicy

# from agent import Agent
from utils import parse_arguments


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
    _, obs = env.reset()
    episode_rewards = []
    i = 0 
    while not done and i < max_iters:
        # RLBench env doesn't have render
        # if render:
        #     env.render()
            
        # _states are only useful when using LSTM policies
        print(obs) 
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
    
    obs_config = ObservationConfig()
    # turn on additional cameras from robot POV, use if end-to-end
    obs_config.set_all(False)  

    action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
    env = Environment(
        action_mode, obs_config=obs_config, headless=not render)

    # TaskEnvironment
    task = env.get_task(ReachTarget)
    task.observation_space = spaces.Box(-np.inf, np.inf, shape=(40,))
    task.action_space = spaces.Box(-1.0, 1.0, shape=(8,))
    task.metadata  = {
            'render.modes': ['human', 'rgb_array'],
            # 'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

    for i in range(100):
        if i % 40 == 0:
            print('Reset Episode')
            descriptions, obs = task.reset()
        action = np.random.randn(env.action_size)
        obs, reward, terminate = task.step(action)

    # agent
    model = PPO(MlpPolicy, task, learning_rate=lr, verbose=0)
    if model_path != "":
        print("Loading Existing model: %s" % model_path)
        model.load(model_path)

    if is_train:
        model.learn(total_timesteps=num_episodes)
        model.save("weights_%d" % timestamp)
    else:
        for i in range(5):
            run_episode(model, task, max_iters=500)

    env.shutdown()


