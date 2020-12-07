import numpy as np
import copy
import ipdb
import os
import collections
import wandb

import torch

from multiprocessing import Pool

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common import logger, utils
from reach_task import ReachTargetCustom
from rlbench.backend.spawn_boundary import SpawnBoundary

from eval_utils import *

import ray
ray.init()

MAML_ID = 0
REPTILE_ID = 1


AvgMetricStore = collections.namedtuple(
    'AvgMetricStore', ['reward', 'success_rate',
                       'entropy_loss', 'pg_loss', 'value_loss', 'loss'])


class MetricStore(object):
    def __init__(self):
        self.total_reward = 0.0
        self.total_success_rate = 0.0
        self.total_entropy_loss = 0.0
        self.total_pg_loss = 0.0
        self.total_value_loss = 0.0
        self.total_loss = 0.0

    def add(self, metrics):
        reward, success_rate, entropy_loss, pg_loss, value_loss, loss = metrics
        self.total_reward += reward
        self.total_success_rate += success_rate
        self.total_entropy_loss += entropy_loss
        self.total_pg_loss += pg_loss
        self.total_value_loss += value_loss
        self.total_loss += loss

    def avg(self, count):
        count = float(count)
        return AvgMetricStore(self.total_reward / count,
                              self.total_success_rate / count,
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
                             base_adapt_kwargs, algo_type):
        # pick a task
        self.model.env.switch_task_wrapper(
            self.model.env, ReachTargetCustom, target_position=target)
        print("Switched to new target:", target)

        # copy over current original weights
        if orig_model_state_dict is not None:
            self.model.policy.load_state_dict(orig_model_state_dict)

        # train new model on K trajectories
        self.model.learn(**base_adapt_kwargs)

        if algo_type == MAML_ID:
            # collect new gradients for a one iteration
            # (NOTE: not one trajectory like paper does, shouldn't make a difference)
            # learn() already calls loss.backward()
            self.model.learn(total_timesteps=1 *
                             self.base_init_kwargs['n_steps'])

        metrics = [self.model.reward, self.model.success_rate, self.model.entropy_loss,
                   self.model.pg_loss, self.model.value_loss, self.model.loss]

        gradients = [p.grad.data for p in self.model.policy.parameters()]
        parameters = [p.data for p in self.model.policy.parameters()]

        return gradients, parameters, metrics

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
        if state_dict is not None:
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

    def learn(self, algo_type, num_iters, save_kwargs):
        utils.configure_logger(
            self.base_init_kwargs["verbose"], save_kwargs["tensorboard_log"], "PPO")

        # log training results
        wandb.init(project="IDL - MAML - Train", entity="idl-project")
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
                a=self.num_tasks, size=self.task_batch_size, replace=False)

            metric_store = MetricStore()

            # run multiple MAML task rollouts in parallel
            orig_model_state_dict = orig_model.state_dict()
            results = ray.get([
                self.model_policy_vec[i].perform_task_rollout.remote(
                    orig_model_state_dict=orig_model_state_dict,
                    target=self.targets[task],
                    base_adapt_kwargs=self.base_adapt_kwargs,
                    algo_type=algo_type)
                for i, task in enumerate(tasks)])

            # initialize gradients
            if algo_type == MAML_ID:
                optimizer.zero_grad()
                for p in orig_model.parameters():
                    p.grad = torch.zeros_like(p).to(device)

                # sum up gradients and store metrics
                for gradients, _, metrics in results:
                    metric_store.add(metrics)
                    for orig_p, grad in zip(orig_model.parameters(), gradients):
                        orig_p.grad += grad / self.task_batch_size
            else:
                # sum up gradients and store metrics
                for i, orig_p in enumerate(orig_model.parameters()):
                    mean_p = sum(res[1][i]
                                 for res in results) / self.task_batch_size
                    # weights = weights_before + lr*(weights_after - weights_before)  <--- grad
                    # weights = weights_before + lr*grad
                    # optimizer: weights = weights_before - lr*grad
                    # optimizer: weights = weights_before + lr*(-grad)
                    # optimizer: weights = weights_before + lr*(weights_before - weights_after)
                    orig_p.grad = orig_p.data - mean_p

                for _, metrics in results:
                    metric_store.add(metrics)

            # apply gradients
            optimizer.step()

            # track performance
            (avg_reward, avg_success_rate, avg_entropy_loss, avg_pg_loss,
                avg_val_loss, avg_loss) = metric_store.avg(self.task_batch_size)
            wandb.log(
                {
                    "mean_reward": avg_reward,
                    "success_rate": avg_success_rate,
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

    def eval_performance(self, model_type, save_kwargs, num_iters=100, targets=None, base_adapt_kwargs=None, model_path=''):
        """Used to compare speed in learning between randomly-initialized weights.
        Runs vanilla PPO using the base model on K fixed tasks, each independent.
        Mean reward is stored for each trial.

        To run, instatiate MAML using model_path='' or model_path='<existing_model>'
        and set task_batch_size = how many CPU cores available to run more tests in parallel.
        Then call eval_performance with some specified targets, or None if the test targets should be
        generated from scratch. Number of test targets should be <= task_batch_size
        just to avoid storing all weights multiple times.

        Args:
            targets ([type], optional): [description]. Defaults to None.
            restore_weights (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
        # load test targets
        if targets is None:
            print("No Test Targets specified. Using default:")
            for v in self.targets:
                print(v)
            targets = self.targets
            num_tasks = self.num_tasks
        else:
            num_tasks = len(targets)
        assert(num_tasks <= self.task_batch_size)

        # load evaluation params for PPO training
        if base_adapt_kwargs is None:
            print("No PPO train args specified. Using default:")
            print(self.base_adapt_kwargs)
            base_adapt_kwargs = self.base_adapt_kwargs

        assert base_adapt_kwargs['total_timesteps'] == self.base_init_kwargs["n_steps"], \
            "We need to collect mean reward and loss at each timestep or epoch, so this must be 1*n_steps!"

        # log Results
        wandb.init(project="IDL - MAML - Eval", entity="idl-project")
        config = {
            "num_tasks": num_tasks,
            "task_batch_size": self.task_batch_size,
            "alpha": self.alpha,
            "beta": self.beta,
            "model_type": model_type}
        wandb.config = config
        utils.configure_logger(
            self.base_init_kwargs["verbose"], save_kwargs["tensorboard_log"], "PPO")

        if save_kwargs["save_targets"]:
            target_path = os.path.join(save_kwargs["save_path"], "targets")
            np.save(target_path, self.targets)

        # load same initial model into all workers
        if model_path != "":
            [self.model_policy_vec[i].load_model.remote(model_path=model_path)
                for i in range(num_tasks)]
        else:
            # use base model and set all other worker models to be same initial weights
            orig_model, device = ray.get(
                self.model_policy_vec[self.BASE_ID].get_model.remote())
            orig_model_state_dict = orig_model.state_dict()

            other_workers = list(range(num_tasks))
            other_workers.pop(self.BASE_ID)
            [self.model_policy_vec[i].load_model.remote(state_dict=orig_model_state_dict)
                for i in other_workers]

        all_metrics = []

        # for num_iters, observe how fast this set of initialized weights can learn each specific task
        for iter in range(num_iters):
            # for each batch of test tasks
            results = ray.get([
                self.model_policy_vec[i].perform_task_rollout.remote(
                    orig_model_state_dict=None,  # keep training existing model
                    target=target,
                    base_adapt_kwargs=base_adapt_kwargs)
                for i, target in enumerate(targets)])

            metric_store = MetricStore()
            for _, metrics in results:
                metric_store.add(metrics)

            # store metrics averaged over all test tasks
            all_metrics.append(metric_store.avg(num_tasks))

            # track performance
            (avg_reward, avg_success_rate, avg_entropy_loss, avg_pg_loss,
                avg_val_loss, avg_loss) = metric_store.avg(self.task_batch_size)
            wandb.log(
                {
                    "mean_reward": avg_reward,
                    "success_rate": avg_success_rate,
                    "entropy_loss": avg_entropy_loss,
                    "policy_gradient_loss": avg_pg_loss,
                    "value_loss": avg_val_loss,
                    "loss": avg_loss
                }
            )

            # save weights every save_freq and at the end
            if (iter > 0 and iter % save_kwargs["save_freq"] == 0) or iter == num_iters-1:
                path = os.path.join(
                    save_kwargs["save_path"], f"{model_type}_{iter}_iters")
                self.model_policy_vec[self.BASE_ID].save.remote(None, path)

        return all_metrics

    def close(self):
        [self.model_policy_vec[i].close.remote()
         for i in range(self.task_batch_size)]
