import gymnasium as gym
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

class ChangeMassWrapper(gym.Wrapper):
    def __init__(self, env, torso_mass=6):
        super().__init__(env)
        self.torso_mass = torso_mass
        # Adjust the torso mass
        self.env.model.body_mass[1] = self.torso_mass

def main():
    # Environment setup
    n_envs = 4
    seed = 42
    torso_mass = 10

    env = make_vec_env('Hopper-v4', n_envs=n_envs, seed=seed, vec_env_cls=SubprocVecEnv,
                       wrapper_class=ChangeMassWrapper, wrapper_kwargs={'torso_mass': torso_mass})
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # PPO setup
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_hopper_tensorboard/")

    # Training
    model.learn(total_timesteps=1e6)

    # Saving the model
    model.save("ppo_hopper")

    # Don't forget to save the VecNormalize statistics when training is finished!
    env.save("ppo_hopper_vec_normalize.pkl")

if __name__ == '__main__':
    main()
