
import numpy as np
from env.grasp_env import GraspEnv
from env.reach_task import ReachTargetCustom

class MultiTaskEnv(GraspEnv):

    test_targets = [
        np.array([0.3, -0.2, 1]),
        np.array([0.3, -0.1, 1]),
        np.array([0.3,  0.0, 1]),
        np.array([0.3,  0.1, 1]),
        np.array([0.3,  0.2, 1]),
    ]

    train_targets = [
        np.array([0.41070282, -0.03492617,  1.0488075]),
        np.array([0.41892517, -0.22750987,  1.061908]),
        np.array([0.14243242, -0.06317406,  0.8563684]),
        np.array([0.38756776, -0.13891503,  1.0362707]),
        np.array([0.33909896,  0.15469943,  1.0564715]),
        np.array([0.15442285, -0.28426912,  1.1886301]),
        np.array([0.42248863,  0.02023844,  1.0861320]),
        np.array([0.14930376,  0.00653961,  0.7940585]),
        np.array([0.15692492,  0.24195677,  1.1235185]),
        np.array([0.14727633, -0.21228942,  1.0185342])
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_task(self, task_num=None, is_train=True):
        
        self.task = self.env.get_task(self.task_class)

        if task_num is None:
            self.task._task.target_position = None
        
        elif is_train:
            assert task_num < len(self.train_targets), "Task requested is grater than total tasks available"
            self.task._task.target_position = self.train_targets[task_num]
        else:
            assert task_num < len(self.test_targets), "Task requested is grater than total tasks available"
            self.task._task.target_position = self.test_targets[task_num]

        self.task.reset()


if __name__ == "__main__":
    
    import time

    env = MultiTaskEnv(task_class=ReachTargetCustom, render_mode="human")

    for i in range(10):
        env.set_task(i)
        time.sleep(1)

    env.close()
