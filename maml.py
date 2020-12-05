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
def perform_task_rollout(model, env, orig_model_state_dict, target,
                         base_adapt_kwargs, base_init_kwargs):
    # pick a task
    model.env.switch_task_wrapper(
        model.env, ReachTargetCustom, target_position=target)
    print("Switched to new target:", target)

    # copy over current original weights
    model.policy.load_state_dict(orig_model_state_dict)

    # train new model on K trajectories
    model.learn(**base_adapt_kwargs)

    # collect new gradients for a one iteration
    # (NOTE: not one trajectory like paper does, shouldn't make a difference)
    # learn() already calls loss.backward()
    model.learn(total_timesteps=1*base_init_kwargs['n_steps'])

    gradients = [p.grad.data for p in model.policy.parameters()]

    metrics = [model.reward, model.entropy_loss,
               model.value_loss, model.loss]

    return gradients, metrics


class MAML(object):
    def __init__(self, BaseAlgo: BaseAlgorithm, EnvClass, num_tasks, task_batch_size,
                 alpha, beta, env_kwargs, base_init_kwargs, base_adapt_kwargs, targets=None):
        """
            BaseAlgo:
            task_envs: [GraspEnv, ...]

            Task-Agnostic because loss function defined by Advantage = Reward - Value function.

        """
        self.BaseAlgo = BaseAlgo
        self.num_tasks = num_tasks
        self.task_batch_size = task_batch_size

        # learning hyperparameters
        self.alpha = alpha
        self.beta = beta

        # self.model = BaseAlgo(learning_rate=alpha, **base_init_kwargs)
        # self.model.env.switch_task_wrapper = base_init_kwargs["env"].switch_task_wrapper
        self.base_init_kwargs = base_init_kwargs
        self.base_adapt_kwargs = base_adapt_kwargs

        # randomly chosen set of static reach tasks
        if targets is None:
            self.targets = []
            assert("NO TASKS!")
            # for _ in range(num_tasks):
            #     [obs] = self.model.env.reset()
            #     target_position = obs[-3:]
            #     self.targets.append(target_position)
        else:
            self.targets = targets

        self.env_vec = [EnvClass(**env_kwargs) for i in range(task_batch_size)]
        self.model_vec = [BaseAlgo(learning_rate=alpha, env=self.env_vec[i], **base_init_kwargs)
                          for i in range(task_batch_size)]

    def learn(self, num_iters, save_kwargs):
        utils.configure_logger(
            self.base_init_kwargs["verbose"], save_kwargs["tensorboard_log"], "PPO")

        if save_kwargs["save_targets"]:
            target_path = os.path.join(save_kwargs["save_path"], "targets")
            np.save(target_path, self.targets)

        # copy set of parameters once
        # arbitrarily pick first model, all randomly initialized
        orig_model = copy.deepcopy(self.model_vec[0].policy)
        optimizer = torch.optim.Adam(orig_model.parameters(), lr=self.beta)
        # lr_scheduler = torch.

        for iter in range(num_iters):
            sum_gradients = [torch.zeros(p.shape).to(self.model.device)
                             for p in orig_model.parameters()]

            # sample task_batch_size tasks from set of [0, num_task) tasks
            tasks = np.random.choice(
                a=self.num_tasks, size=self.task_batch_size)

            rewards = []
            entropy_losses = []
            pg_losses = []
            value_losses = []
            losses = []

            orig_model_state_dict = orig_model.state_dict()
            results = [
                perform_task_rollout.remote(
                    model=self.model_vec[i], env=self.env_vec[i],
                    orig_model_state_dict=orig_model_state_dict,
                    target=self.targets[task],
                    base_adapt_kwargs=self.base_adapt_kwargs,
                    base_init_kwargs=self.base_init_kwargs)

                for i, task in enumerate(tasks)]

            import ipdb
            ipdb.set_trace()
            results = ray.get(results)

            for gradients, metrics in results:
                for orig_p, grad in zip(orig_model.parameters(), gradients):
                    orig_p.grad += grad

            # apply sum of gradients to original model
            # no need for optimizer.zero_grad() because gradients directly set, not accumulated

            optimizer.step()

            if iter > 0 and iter % save_kwargs["save_freq"] == 0:
                path = os.path.join(save_kwargs["save_path"], f"{iter}_iters")
                self.model.save(path)

            # log Results
            # logger.record("train/mean_reward", np.mean(rewards))
            # logger.record("train/entropy_loss", np.mean(entropy_losses))
            # logger.record("train/policy_gradient_loss", np.mean(pg_losses))
            # logger.record("train/value_loss", np.mean(value_losses))
            # logger.record("train/loss", np.mean(losses))

        # set final weights back into model
        self.model.policy.load_state_dict(orig_model.state_dict())

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
