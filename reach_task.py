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


class ReachTargetCustom(ReachTarget):

    def reward(self) -> float:

        dist = self.target.get_position(self.robot.arm.get_tip())
        dist_val = np.linalg.norm(dist) / 10.0
#         if dist_val < 0.1:
#             return 1
        return -dist_val

    def get_name(self):
        return "reach_target"


class ReachTargetCustomStatic(ReachTarget):

    def reward(self) -> float:

        dist = self.target.get_position(self.robot.arm.get_tip())
        dist_val = np.linalg.norm(dist) / 10.0
#         if dist_val < 0.1:
#             return 1
        return -dist_val

    def get_name(self):
        return "reach_target"

    def init_task(self) -> None:

        super().init_task()
        # set object poses only once in the beginning
        b = SpawnBoundary([self.boundaries])
        self.static_positions = []
        for ob in [self.target, self.distractor0, self.distractor1]:
            b.sample(ob, min_distance=0.2,
                     min_rotation=(0, 0, 0), max_rotation=(0, 0, 0))
            self.static_positions.append(ob.get_position())

        print("static positions:")
        for v in self.static_positions:
            print(v)

        import ipdb
        ipdb.set_trace()

    def init_episode(self, index: int) -> List[str]:

        color_name, color_rgb = colors[index]
        self.target.set_color(color_rgb)
        print("static positions:")
        for v in self.static_positions:
            print(v)
        for i, ob in enumerate([self.target, self.distractor0, self.distractor1]):
            ob.set_position(self.static_positions[i])

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
