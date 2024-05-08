import gymnasium as gym

from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO

class ChangeMassWrapper(gym.Wrapper):
    def __init__(self, env, torso_mass=6):
        super().__init__(env)
        self.torso_mass = torso_mass
        self.env.model.body_mass[1] = self.torso_mass

# Set your desired number of environments, seed, and torso mass
n_envs = 4  # Example: 4 parallel environments
seed = 42   # Example: Random seed for reproducibility
torso_mass = 10  # Example: New torso mass for the Hopper

env = make_vec_env('Hopper-v4', n_envs=n_envs, seed=seed, vec_env_cls=SubprocVecEnv,
                   wrapper_class=ChangeMassWrapper, wrapper_kwargs={'torso_mass': torso_mass})
env = VecNormalize(env, norm_obs=True, norm_reward=True)

#  Train a DRL model of your choice (do justify your choice) on the Hopper-v4 environment, choosing the rigth algorithm and hyperparameters

# PPO setup
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_hopper_tensorboard/")

# Training
model.learn(total_timesteps=1e6)

# Saving the model
model.save("ppo_hopper")

# Don't forget to save the VecNormalize statistics when training is finished!
env.save("ppo_hopper_vec_normalize.pkl")