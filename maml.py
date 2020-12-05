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


class MetricStore(object):
    def __init__(self):
        self.total_reward = 0.0
        self.total_entropy_loss = 0.0
        self.total_pg_loss = 0.0
        self.total_value_loss = 0.0
        self.total_loss = 0.0

    def add(self, metrics):
        reward, entropy_loss, pg_loss, value_loss, loss = metrics
        self.total_reward += reward
        self.total_entropy_loss += entropy_loss
        self.total_pg_loss += pg_loss
        self.total_value_loss += value_loss
        self.total_loss += loss

    def avg(self, count):
        count = float(count)
        return (self.total_reward / count,
                self.total_entropy_loss / count,
                self.total_pg_loss / count,
                self.total_value_loss / count,
                self.total_loss / count)


@ray.remote
class MAML_Worker(object):
    def __init__(self, EnvClass, ModelClass, env_kwargs, model_kwargs):
        self.env = EnvClass(**env_kwargs)
        self.model = ModelClass(env=self.env, **model_kwargs)
        self.model.env.switch_task_wrapper = self.env.switch_task_wrapper
        self.base_init_kwargs = model_kwargs

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
        pre_metrics = [self.model.reward, self.model.entropy_loss, self.model.pg_loss,
                       self.model.value_loss, self.model.loss]

        # collect new gradients for a one iteration
        # (NOTE: not one trajectory like paper does, shouldn't make a difference)
        # learn() already calls loss.backward()
        self.model.learn(total_timesteps=1*self.base_init_kwargs['n_steps'])

        gradients = [p.grad.data for p in self.model.policy.parameters()]

        post_metrics = [self.model.reward, self.model.entropy_loss, self.model.pg_loss,
                        self.model.value_loss, self.model.loss]

        return gradients, post_metrics, pre_metrics

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

    def save(self, state_dict, save_path):
        self.load_model(state_dict=state_dict)
        self.model.save(save_path)

    def close(self):
        self.env.close()


class MAML(object):
    BASE_ID = 0

    def __init__(self, BaseAlgo: BaseAlgorithm, EnvClass, num_tasks, task_batch_size,
                 alpha, beta, model_path, env_kwargs, base_init_kwargs, base_adapt_kwargs, targets=None):
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

        self.base_init_kwargs = base_init_kwargs
        self.base_adapt_kwargs = base_adapt_kwargs

        self.model_policy_vec = [
            MAML_Worker.remote(EnvClass=EnvClass, ModelClass=BaseAlgo, env_kwargs=env_kwargs,
                               model_kwargs=base_init_kwargs)
            for i in range(task_batch_size)]

        # optional load existing model
        self.model_path = model_path
        if model_path != "":
            print("Loading Existing model: %s" % model_path)
            self.model_policy_vec[self.BASE_ID].load_model.remote(
                model_path=model_path)
        else:
            print("No Existing model. Randomly initializing weights")

        # randomly chosen set of static reach tasks
        if targets is None:
            self.targets = []
            for _ in range(num_tasks):
                [obs] = ray.get(
                    self.model_policy_vec[self.BASE_ID].sample_task.remote())

                target_position = obs[-3:]
                self.targets.append(target_position)
                print(target_position)
        else:
            self.targets = targets

    def learn(self, num_iters, save_kwargs):
        utils.configure_logger(
            self.base_init_kwargs["verbose"], save_kwargs["tensorboard_log"], "PPO")

        # log training results
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

        # initialize base model and optimizer
        orig_model, device = ray.get(
            self.model_policy_vec[self.BASE_ID].get_model.remote())
        optimizer = torch.optim.Adam(orig_model.parameters(), lr=self.beta)
        # lr_scheduler = torch.

        for iter in range(num_iters):
            # sample task_batch_size tasks from set of [0, num_task) tasks
            tasks = np.random.choice(
                a=self.num_tasks, size=self.task_batch_size)

            metric_store = MetricStore()

            # run multiple MAML task rollouts in parallel
            orig_model_state_dict = orig_model.state_dict()
            results = ray.get([
                self.model_policy_vec[i].perform_task_rollout.remote(
                    orig_model_state_dict=orig_model_state_dict,
                    target=self.targets[task],
                    base_adapt_kwargs=self.base_adapt_kwargs)
                for i, task in enumerate(tasks)])

            # initialize gradients
            optimizer.zero_grad()
            for p in orig_model.parameters():
                p.grad = torch.zeros_like(p).to(device)

            # sum up gradients and store metrics
            for gradients, metrics, _ in results:
                metric_store.add(metrics)
                for orig_p, grad in zip(orig_model.parameters(), gradients):
                    orig_p.grad += grad / self.task_batch_size

            # apply gradients
            optimizer.step()

            # track performance
            (avg_reward, avg_entropy_loss, avg_pg_loss,
                avg_val_loss, avg_loss) = metric_store.avg(self.task_batch_size)
            wandb.log(
                {
                    "mean_reward": avg_reward,
                    "entropy_loss": avg_entropy_loss,
                    "policy_gradient_loss": avg_pg_loss,
                    "value_loss": avg_val_loss,
                    "loss": avg_loss
                }
            )

            # save weights every save_freq and at the end
            if (iter > 0 and iter % save_kwargs["save_freq"] == 0) or iter == num_iters-1:
                path = os.path.join(save_kwargs["save_path"], f"{iter}_iters")
                self.model_policy_vec[self.BASE_ID].save.remote(
                    orig_model.state_dict(), path)

    def eval_performance(self, targets=None, restore_weights=True):
        if targets is None:
            targets = self.targets
            num_tasks = self.num_tasks
        else:
            num_tasks = len(targets)
        num_batches = np.ceil(num_tasks / self.task_batch_size).astype(int)

        pre_metric_store = MetricStore()
        post_metric_store = MetricStore()

        orig_model, device = ray.get(
            self.model_policy_vec[self.BASE_ID].get_model.remote())
        orig_model_state_dict = orig_model.state_dict()

        for bi in range(num_batches):
            start_idx = bi * self.task_batch_size
            end_idx = min((bi+1)*self.task_batch_size, len(targets))
            batch_targets = targets[start_idx:end_idx]

            results = ray.get([
                self.model_policy_vec[i].perform_task_rollout.remote(
                    orig_model_state_dict=orig_model_state_dict,
                    target=target,
                    base_adapt_kwargs=self.base_adapt_kwargs)
                for i, target in enumerate(batch_targets)])

            for _, post_metrics, pre_metrics in results:
                post_metric_store.add(post_metrics)
                pre_metric_store.add(pre_metrics)

        # restore original model parameters so next call to eval uses same pre-loaded weights
        if restore_weights:
            assert(self.model_path != "")
            self.model_policy_vec[self.BASE_ID].load_model.remote(
                model_path=self.model_path)
            # NOTE: below doesn't work due to this error:
            # # RuntimeError: Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment
            # self.model_policy_vec[self.BASE_ID].load_model.remote(
            #     state_dict=orig_model_state_dict)

        return pre_metric_store.avg(num_tasks), post_metric_store.avg(num_tasks)

    def close(self):
        [self.model_policy_vec[i].close.remote()
         for i in range(self.task_batch_size)]
