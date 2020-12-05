
import numpy as np
import time
from grasp_env import GraspEnv
from reach_task import ReachTargetCustom
import ray


def wrapper(fn):
    def result(*args, **kwargs):
        return fn.remote(*args, **kwargs)
    return result


def wrap_methods(cls, wrapper):
    functions = set(["set_task"])
    for key, value in cls.__dict__.items():
        # if hasattr(value, '__call__'):
        # print(key)
        if key in functions:
            setattr(cls, key, wrapper(value))


@ray.remote
class MultiTaskEnv(GraspEnv):
    def __init__(self, num_tasks, *args, **kwargs):
        print(args)
        print()
        print(kwargs)
        super().__init__(*args, **kwargs)

        self.num_tasks = num_tasks
        self.set_task = wrapper(self.set_task)

        assert self.num_tasks == 5, "Currently only supports 5 task"

        self.targets = [
            np.array([0.3, -0.2, 1]),
            np.array([0.3, -0.1, 1]),
            np.array([0.3,  0.0, 1]),
            np.array([0.3,  0.1, 1]),
            np.array([0.3,  0.2, 1]),
        ]

        wrap_methods(MultiTaskEnv, wrapper)

    def set_task(self, task_num):

        assert task_num < len(
            self.targets), "Task requested is grater than total tasks available"

        self.task = self.env.get_task(self.task_class)
        self.task._task.target_position = self.targets[task_num]
        self.task.reset()


if __name__ == "__main__":

    env = MultiTaskEnv(
        num_tasks=5, task_class=ReachTargetCustom, render_mode="human")

    for i in range(5):
        env.set_task(i)
        time.sleep(1)

    env.close()
