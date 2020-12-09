import copy
import numpy as np
import torch
from env.grasp_env import GraspEnv
from stable_baselines3 import PPO
import ray

class BaseWorker():
    def __init__(self, config, env_class=GraspEnv, model_class=PPO, **_):
        self.env = env_class(**config)
        self.model = model_class(env=self.env, **config)

    def sample_task(self):
        return self.model.env.reset()

    def get_model(self):
        return copy.deepcopy(self.model.policy), self.model.device

    def load_model(self, state_dict=None, model_path=None):
        if state_dict is not None:
            self.model.policy.load_state_dict(state_dict)
        else:
            assert(model_path is not None)
            self.model = self.model.load(model_path, env=self.env)
            self.model.env.switch_task_wrapper = self.env.switch_task_wrapper

    def set_task(self, task_num):
        self.model.env.envs[0].set_task(task_num)

    def save(self, state_dict, save_path):
        if state_dict is not None:
            self.load_model(state_dict=state_dict)
        self.model.save(save_path)

    def run_eval_eposide(self, max_iters=200):
        done = False
        obs = self.env.reset()
        
        self.model.policy.eval()
        
        episode_rewards = []
        i = 0
        with torch.no_grad():
            while not done and i < max_iters:
                action, _states = self.model.predict(obs)
                obs, reward, done, desc = self.env.step(action)
                episode_rewards.append(reward)
                i += 1
                    
        final_done = desc["is_success"]

        return episode_rewards, final_done
    
    def evaluate(self, num_episodes=5, max_iters=200):

        all_episode_rewards = []
        success = []

        for i in range(num_episodes):
            episode_rewards, final_done = self.run_eval_eposide(max_iters)
            total_reward = sum(episode_rewards)
            all_episode_rewards.append(total_reward)
            success.append(final_done)

        mean_reward = np.mean(all_episode_rewards)
        std_reward = np.std(all_episode_rewards)
        success_rate = sum(success)/len(success)

        return mean_reward, std_reward, success_rate

    def close(self):
        self.env.close()


class WorkerHandler(object):

    def __init__(self, config, worker_class=BaseWorker, num_workers=5, **_):
        
        self.worker_class = worker_class
        self.num_workers = num_workers
        worker_kwargs={

        }
        self.worker_kwargs = worker_kwargs

        self.workers = [ self.worker_class.remote(config, **config) for i in range(self.num_workers) ]
    
    def evaluate(self, num_episodes=5, max_iters=200):
        results = ray.get(
            [ worker.evaluate.remote(num_episodes=num_episodes, max_iters=max_iters) 
                for worker in self.workers ]
        )
    
    def set_task(self, new_tasks):
        results = ray.get(
            [ worker.set_task.remote(new_tasks[i]) 
                for i, worker in enumerate(self.workers) ]
        )
