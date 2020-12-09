from stable_baselines3.ppo.policies import MlpPolicy

class CustomPolicy(MlpPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                net_arch=[64, 64, dict(pi=[64, 64], vf=[64, 64])])
    