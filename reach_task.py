from rlbench.tasks import ReachTarget
import numpy as np
from rlbench.tasks import reach_target


class ReachTargetCustom(ReachTarget):

    def reward(self) -> float:
    
        dist = self.target.get_position(self.robot.arm.get_tip())
        dist_val = 5 * np.linalg.norm(dist) ** 2
        # if dist_val < 0.1:
        #     return 10
        return -dist_val

    def get_name(self):
        return "reach_target"