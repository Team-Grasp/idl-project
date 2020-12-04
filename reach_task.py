from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.const import colors
from rlbench.backend.task import Task
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.conditions import DetectedCondition
from rlbench.tasks import ReachTarget
import numpy as np

import ipdb


class ReachTargetCustom(ReachTarget):
    def reward(self) -> float:
        dist = self.target.get_position(self.robot.arm.get_tip())
        dist_val = np.linalg.norm(dist) / 10.0
#         if dist_val < 0.1:
#             return 1
        return -dist_val

    def get_name(self):
        return "reach_target"


class ReachTargetCustomStatic(ReachTargetCustom):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_position = None

    def init_episode(self, index: int) -> List[str]:
        # vary the colors of distractors
        color_name, color_rgb = colors[index]
        self.target.set_color(color_rgb)

        # set position of objects
        b = SpawnBoundary([self.boundaries])
        # distractors aren't used so just arbitrarily reset position
        # b.sample(self.distractor0, min_distance=0.2,
        #          min_rotation=(0, 0, 0), max_rotation=(0, 0, 0))
        # b.sample(self.distractor1, min_distance=0.2,
        #          min_rotation=(0, 0, 0), max_rotation=(0, 0, 0))

        if self.target_position is None:
            b.sample(self.target, min_distance=0.2,
                     min_rotation=(0, 0, 0), max_rotation=(0, 0, 0))

        else:
            self.target.set_position(self.target_position)

        # randomize the distractor object colors
        # color_choices = np.random.choice(
        #     list(range(index)) + list(range(index + 1, len(colors))),
        #     size=2, replace=False)
        # for ob, i in zip([self.distractor0, self.distractor1], color_choices):
        #     name, rgb = colors[i]
        #     ob.set_color(rgb)

        # randomize pose of objects
        # b = SpawnBoundary([self.boundaries])
        # for ob in [self.target, self.distractor0, self.distractor1]:
        #     b.sample(ob, min_distance=0.2,
        #              min_rotation=(0, 0, 0), max_rotation=(0, 0, 0))
