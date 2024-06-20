import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from typing import Callable
import torch

# from sb3 docs
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

class ChangeMassWrapper(gym.Wrapper):
    def __init__(self, env, torso_mass=6):
        super().__init__(env)
        self.torso_mass = torso_mass
        # Adjust the torso mass
        self.env.model.body_mass[1] = self.torso_mass

def train_model(seed, torso_mass):
    # Environment setup
    n_envs = 1
    env = make_vec_env('Hopper-v4', n_envs=n_envs, seed=seed, vec_env_cls=SubprocVecEnv,
                       wrapper_class=ChangeMassWrapper, wrapper_kwargs={'torso_mass': torso_mass})
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Logging setup
    tensorboard_path = f"./ppo_hopper_tensorboard/seed_{seed}_mass_{torso_mass}"
    new_logger = configure(tensorboard_path, ["stdout", "csv", "tensorboard"])

    total_timesteps = 1e6

    # PPO model instantiation with default parameters
    model = PPO("MlpPolicy", env, verbose=1)

    model.set_logger(new_logger)

    # Training
    model.learn(total_timesteps=total_timesteps)  

    # Save the model and normalization stats
    model.save(f"ppo_hopper_seed_{seed}_mass_{torso_mass}_hopperv3params")
    env.save(f"ppo_hopper_vec_normalize_seed_{seed}_mass_{torso_mass}_hopperv3params.pkl")

    # access stats from the training by tensorboard --logdir=./ppo_hopper_tensorboard/

    # Cleanup environment
    env.close()

def main():
    seeds = [1, 2, 3, 4, 5]
    torso_masses = [3, 6, 9]
    
    # Perform training with different combinations of seeds and torso weights
    for seed in seeds:
        for mass in torso_masses:
            print(f"PPO Training with seed {seed} and torso mass {mass}")
            train_model(seed, mass)
           

if __name__ == '__main__':
    main()
