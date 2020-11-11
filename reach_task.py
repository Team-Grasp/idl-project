from rlbench.tasks import ReachTarget
import numpy as np
from rlbench.tasks import reach_target


class ReachTargetCustom(ReachTarget):

    def reward(self) -> float:
    
        dist = self.target.get_position(self.robot.arm.get_tip())
        dist_val = np.linalg.norm(dist) 
        # if dist_val < 0.1:
        #     return 10
        return -dist_val

    def get_name(self):
        return "reach_target"