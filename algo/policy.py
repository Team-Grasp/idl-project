from stable_baselines3.ppo.policies import MlpPolicy

class CustomPolicy(MlpPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                net_arch=[12, 12, dict(pi=[12, 12], vf=[12, 12])])
    