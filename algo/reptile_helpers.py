import numpy as np
import ray
from env.multitask_env import MultiTaskEnv
import wandb

from algo.base_classes import BaseWorker, WorkerHandler
from algo.utils import *

ray.init()


@ray.remote
class ReptileWorker(BaseWorker):
    def __init__(self, config, **_):
        super().__init__(config, **config)
    
    def learn(self, config, **_):

        assert False, "Implement this"

        self.model.policy.train()

        self.model.learn(**config)
    
        metrics = [self.model.reward, self.model.success_rate, self.model.entropy_loss,
                   self.model.pg_loss, self.model.value_loss, self.model.loss]

        gradients = [p.grad.data for p in self.model.policy.parameters()]
        parameters = [p.data for p in self.model.policy.parameters()]

        return RolloutResults(gradients=gradients, parameters=parameters, metrics=metrics)

class ReptileWorkerHandler(WorkerHandler):
    def __init__(self, config, model_path, base_path="", max_tasks=5, num_workers=5, **_):
        
        assert num_workers <= max_tasks

        super().__init__(config, **config)
        self.base_path = base_path

        if model_path != "":
            self.load_model(model_path=model_path)

    def learn(self, config, num_iters=100, save_freq=1, eval_freq=1, is_wandb=False, **_):
        
        assert False, "Implement this"

        tasks = np.arange(self.num_workers)
        self.set_task(tasks)
        
        for iter in range(num_iters):
        
            results = ray.get(
                [ worker.learn.remote(config, **config) for worker in self.workers ]
            )

            self.process_learn_results(results, is_wandb=is_wandb)

            if iter % eval_freq == 0:
                self.evaluate(is_wandb=is_wandb)

            if iter % save_freq == 0:
                self.save_model(save_path=self.base_path + f"model_{iter}")
        
        return True

    def evaluate(self, num_episodes=5, max_iters=200, is_wandb=False):
        results_eval = ray.get(
            [ worker.evaluate.remote(num_episodes=num_episodes, max_iters=max_iters) 
                for worker in self.workers ]
        )

        self.process_eval_results(results_eval, is_wandb=is_wandb)

    def process_learn_results(self, results, is_wandb=False):
        
        assert False, "Implement this"
        
        metric_store = MetricStore()

        for res in results:
            metric_store.add(res.metrics)

        (avg_reward, avg_success_rate, avg_entropy_loss, avg_pg_loss,
            avg_val_loss, avg_loss) = metric_store.avg(self.num_workers)
        
        if is_wandb:
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

    def process_eval_results(self, results_eval, is_wandb=False):
        
        all_mean_rewards = [result[0] for result in results_eval]
        all_std_rewards = [result[1] for result in results_eval]
        all_success_rate = [result[2] for result in results_eval]

        avg_mean_reward_eval = sum(all_mean_rewards)/ len(all_mean_rewards)
        avg_std_reward_eval = sum(all_std_rewards)/ len(all_std_rewards)
        avg_success_rate_eval = sum(all_success_rate)/ len(all_success_rate)

        log_obj = {
            "mean_reward_eval": avg_mean_reward_eval,
            "std_reward_eval": avg_std_reward_eval,
            "success_rate_eval": avg_success_rate_eval,
        }
        
        print(log_obj)

        if is_wandb:
            wandb.log(log_obj)
