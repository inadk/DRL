import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
import optuna
import time
from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

class ChangeMassWrapper(gym.Wrapper):
    def __init__(self, env, torso_mass=6):
        super().__init__(env)
        self.torso_mass = torso_mass
        self.env.model.body_mass[1] = self.torso_mass

def get_episode_rewards(envs):
    rewards = []
    for env in envs:
        try:
            rewards.extend(env.get_attr("episode_rewards"))
        except AttributeError:
            print("AttributeError: 'episode_rewards' not found.")
    return rewards

def train_model(trial, seed, torso_mass):
    learning_rate = trial.suggest_float("learning_rate", 5e-4, 1e-3)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    gamma = trial.suggest_float("gamma", 0.98, 0.9999)
    n_steps = trial.suggest_int("n_steps", 256, 2048)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    n_envs = 4
    try:
        env = make_vec_env('Hopper-v4', n_envs=n_envs, seed=seed, vec_env_cls=SubprocVecEnv,
                           wrapper_class=ChangeMassWrapper, wrapper_kwargs={'torso_mass': torso_mass})
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
    except Exception as e:
        print(f"Error creating environment: {e}")
        return None, 0

    tensorboard_path = f"./ppo_hopper_tensorboard/seed_{seed}_mass_{torso_mass}"
    new_logger = configure(tensorboard_path, ["stdout", "csv", "tensorboard"])

    model = PPO("MlpPolicy", env, verbose=1,
                batch_size=batch_size,
                n_steps=n_steps,
                learning_rate=linear_schedule(learning_rate),
                tensorboard_log=tensorboard_path,
                clip_range=linear_schedule(clip_range),
                gamma=gamma)

    model.set_logger(new_logger)

    start_time = time.time()
    try:
        model.learn(total_timesteps=int(1e4))
        duration = time.time() - start_time

        model.save(f"ppo_hopper_seed_{seed}_mass_{torso_mass}_annealing_lr")
        env.save(f"ppo_hopper_vec_normalize_seed_{seed}_mass_{torso_mass}_annealing_lr.pkl")
        
        total_rewards = get_episode_rewards(env.get_attr("env"))
        env.close()
        if not total_rewards:
            print(f"No rewards were collected during training.")
        return total_rewards, duration
    except Exception as e:
        print(f"Error during training: {e}")
        env.close()
        return None, 0

def objective(trial):
    seed = trial.suggest_categorical("seed", [1, 2, 3])
    torso_mass = trial.suggest_categorical("torso_mass", [3, 6, 9])
    total_rewards, duration = train_model(trial, seed, torso_mass)
    if total_rewards is None or not total_rewards:
        return float('-inf')
    trial.set_user_attr("duration", duration)
    return sum(total_rewards) / len(total_rewards)

def print_progress(study, trial):
    completed_trials = len(study.trials)
    if completed_trials > 1:
        average_duration = sum([t.user_attrs["duration"] for t in study.trials if "duration" in t.user_attrs]) / completed_trials
        remaining_trials = study.n_trials - completed_trials
        estimated_time_remaining = average_duration * remaining_trials
        print(f"Progress: {completed_trials}/{study.n_trials} trials completed.")
        print(f"ETA: {estimated_time_remaining/60:.2f} minutes remaining.")

def main():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, callbacks=[print_progress])

    print('Best trial:', study.best_trial.params)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    main()
