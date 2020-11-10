from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget

import numpy as np
import math
import matplotlib.pyplot as plt

# import numpy as np
# import time
# DO NOT IMPORT THIS HERE!!! CAUSES WEIRD QT ERROR
# from stable_baselines3 import PPO

# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.ppo.policies import MlpPolicy

# from agent import Agent
# from utils import parse_arguments


class Agent(object):
    def __init__(self, action_size):
        self.action_size = action_size

    def act(self, obs):
        # arm = np.random.normal(0.0, 0.1, size=(self.action_size - 1,))
        print("joint pos")
        print(obs.joint_positions)
        print("gripper pose:")
        print(obs.gripper_pose)
        arm = np.zeros(self.action_size-1)
        arm[0] = 5 * math.pi / 180.0
        # arm[0:3] = [0, 0, 0]
        # arm[3:] = [0, 0, 0, 1]
        gripper = [1.0]  # Always open
        print()
        return np.concatenate([arm, gripper], axis=-1)

if __name__ == "__main__":
    # import gym
    # import rlbench.gym
    full_pos = np.load("full_pos.npy")
    plt.plot(full_pos[:, 0], label="x")
    plt.plot(full_pos[:, 1], label="y")
    plt.plot(full_pos[:, 2], label="z")
    plt.legend()
    plt.show()
    exit()

    # temp_env = gym.make('reach_target-state-v0')
    # print(temp_env.action_space)
    
    obs_config = ObservationConfig()
    obs_config.set_all(False)
    obs_config.joint_positions = True
    obs_config.gripper_pose = True

    action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
    env = Environment(
        action_mode, obs_config=obs_config, headless=False)
    # env.launch()  # not needed, called by env.get_task()

    task = env.get_task(ReachTarget)

    agent = Agent(env.action_size)

    training_steps = 200
    episode_length = 200
    obs = None
    full_pos = []
    for i in range(training_steps):
        if i % episode_length == 0:
            print('Reset Episode')
            descriptions, obs = task.reset()
            print(descriptions)
        action = agent.act(obs)
        print(task.get_observation().joint_positions)
        print("Action:")
        print(action)
        obs, reward, terminate = task.step(action)
        full_pos.append(obs.joint_positions[:3])

    full_pos = np.vstack(full_pos)
    np.save("full_pos", full_pos)

    print('Done')
    env.shutdown()


