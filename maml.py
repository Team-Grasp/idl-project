import numpy as np
import copy
import ipdb
import os

import torch

from multiprocessing import Pool

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common import logger, utils
from reach_task import ReachTargetCustom
from rlbench.backend.spawn_boundary import SpawnBoundary

from eval_utils import *

import ray
ray.init()


@ray.remote
class MultiTaskEnvRay(object):
    def __init__(self, EnvClass, ModelClass, env_kwargs, model_kwargs):
        env = EnvClass(**env_kwargs)
        self.model = ModelClass(env=env, **model_kwargs)
        self.model.env.switch_task_wrapper = env.switch_task_wrapper
        self.base_init_kwargs = model_kwargs

    # def get_gradients(self):

    # def get_metrics(self)

    def perform_task_rollout(self, orig_model_state_dict, target,
                             base_adapt_kwargs):
        # pick a task
        self.model.env.switch_task_wrapper(
            self.model.env, ReachTargetCustom, target_position=target)
        print("Switched to new target:", target)

        # copy over current original weights
        self.model.policy.load_state_dict(orig_model_state_dict)

        # train new model on K trajectories
        self.model.learn(**base_adapt_kwargs)

        # collect new gradients for a one iteration
        # (NOTE: not one trajectory like paper does, shouldn't make a difference)
        # learn() already calls loss.backward()
        self.model.learn(total_timesteps=1*self.base_init_kwargs['n_steps'])

        gradients = [p.grad.data for p in self.model.policy.parameters()]

        metrics = [self.model.reward, self.model.entropy_loss,
                   self.model.value_loss, self.model.loss]

        return gradients, metrics

    def sample_task(self):
        return self.model.env.reset()

    def get_model(self):
        return copy.deepcopy(self.model.policy), self.model.device

    def save(self, state_dict, path):
        self.model.policy.load_state_dict(state_dict)
        self.model.save(path)


class MAML(object):
    def __init__(self, BaseAlgo: BaseAlgorithm, EnvClass, num_tasks, task_batch_size,
                 alpha, beta, env_kwargs, base_init_kwargs, base_adapt_kwargs, targets=None):
        """
            BaseAlgo:
            task_envs: [GraspEnv, ...]

            Task-Agnostic because loss function defined by Advantage = Reward - Value function.

        """
        self.num_tasks = num_tasks
        self.task_batch_size = task_batch_size

        # learning hyperparameters
        self.alpha = alpha
        self.beta = beta

        # self.model = BaseAlgo(learning_rate=alpha, **base_init_kwargs)
        # self.model.env.switch_task_wrapper = base_init_kwargs["env"].switch_task_wrapper
        self.base_init_kwargs = base_init_kwargs
        self.base_adapt_kwargs = base_adapt_kwargs

        self.model_policy_vec = [
            MultiTaskEnvRay.remote(EnvClass=EnvClass, ModelClass=BaseAlgo, env_kwargs=env_kwargs,
                                   model_kwargs=base_init_kwargs)
            for i in range(task_batch_size)]

        # randomly chosen set of static reach tasks
        if targets is None:
            self.targets = []
            for _ in range(num_tasks):
                [obs] = ray.get(self.model_policy_vec[0].sample_task.remote())
                target_position = obs[-3:]
                print(target_position)
                self.targets.append(target_position)
        else:
            self.targets = targets

    def learn(self, num_iters, save_kwargs):
        utils.configure_logger(
            self.base_init_kwargs["verbose"], save_kwargs["tensorboard_log"], "PPO")

        import wandb
        wandb.init(project="IDL - MAML", entity="idl-project")
        config = {
            "num_tasks": self.num_tasks,
            "task_batch_size": self.task_batch_size,
            "alpha": self.alpha,
            "beta": self.beta
        }
        wandb.config = config

        if save_kwargs["save_targets"]:
            target_path = os.path.join(save_kwargs["save_path"], "targets")
            np.save(target_path, self.targets)

        # copy set of parameters once
        # arbitrarily pick first model, all randomly initialized
        # orig_model = BaseAlgo(env=env)
        orig_model, device = ray.get(
            self.model_policy_vec[0].get_model.remote())
        optimizer = torch.optim.Adam(orig_model.parameters(), lr=self.beta)

        # lr_scheduler = torch.

        for iter in range(num_iters):
            # sample task_batch_size tasks from set of [0, num_task) tasks
            tasks = np.random.choice(
                a=self.num_tasks, size=self.task_batch_size)

            total_reward = 0.0
            total_entropy_loss = 0.0
            total_pg_loss = 0.0
            total_value_loss = 0.0
            total_loss = 0.0

            orig_model_state_dict = orig_model.state_dict()
            results = ray.get([
                self.model_policy_vec[i].perform_task_rollout.remote(
                    orig_model_state_dict=orig_model_state_dict,
                    target=self.targets[task],
                    base_adapt_kwargs=self.base_adapt_kwargs)
                for i, task in enumerate(tasks)])

            optimizer.zero_grad()
            for p in orig_model.parameters():
                p.grad = torch.zeros_like(p).to(device)

            for gradients, metrics in results:
                reward, entropy_loss, value_loss, loss = metrics
                total_reward += reward
                total_entropy_loss += entropy_loss
                total_value_loss += value_loss
                total_loss += loss

                for orig_p, grad in zip(orig_model.parameters(), gradients):
                    orig_p.grad += grad

            optimizer.step()

            wandb.log(
                {
                    "mean_reward": total_reward / self.task_batch_size,
                    "entropy_loss": total_entropy_loss / self.task_batch_size,
                    "policy_gradient_loss": total_pg_loss / self.task_batch_size,
                    "value_loss": total_value_loss / self.task_batch_size,
                    "loss": total_loss / self.task_batch_size
                }
            )

            if iter > 0 and iter % save_kwargs["save_freq"] == 0:
                path = os.path.join(save_kwargs["save_path"], f"{iter}_iters")
                self.model_policy_vec[0].save.remote(
                    orig_model.state_dict(), path)

        # set final weights back into model
        # self.model.policy.load_state_dict(orig_model.state_dict())

    def eval_performance(self, target_position, restore_weights=True):
        # save original model weights
        orig_model = copy.deepcopy(self.model.policy)

        # load original weights
        self.model.policy.load_state_dict(orig_model.state_dict())

        # set task
        try:
            self.model.env.switch_task_wrapper(
                self.model.env, ReachTargetCustom, target_position=target_position)
        except AttributeError:
            self.model.env.switch_task_wrapper = self.base_init_kwargs["env"].switch_task_wrapper
            self.model.env.switch_task_wrapper(
                self.model.env, ReachTargetCustom, target_position=target_position)

        # run an episode of evaluation
        print("Pre-evaluation...")
        pre_adapt_rewards = run_episode(
            self.model, self.model.env, max_iters=200)

        # run one iteration of training on this task
        print("Adapting...")
        self.model.learn(**self.base_adapt_kwargs)

        # run another episode of evaluation to see how much reward improved
        print("Post-evaluation...")
        post_adapt_rewards = run_episode(
            self.model, self.model.env, max_iters=200)

        # optionally restore original weights
        if restore_weights:
            self.model.policy.load_state_dict(orig_model.state_dict())

        return pre_adapt_rewards, post_adapt_rewards

    def predict(self, obs):
        return self.model.predict(obs)
