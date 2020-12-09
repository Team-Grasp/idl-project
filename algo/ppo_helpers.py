import ray
from algo.base_classes import BaseWorker, WorkerHandler

ray.init()

@ray.remote
class PPOWorker(BaseWorker):
    def __init__(self, config, **_):
        super().__init__(config, **config)

class PPOWorkerHandler(WorkerHandler):
    def __init__(self, config, base_path="", **_):
        super().__init__(config, **config)
        self.base_path = base_path
    
    def load_model(self):
        pass        



