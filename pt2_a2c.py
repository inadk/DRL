import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

# Create environment
env = make_vec_env('BipedalWalker-v3', n_envs=1)
env = VecNormalize(env, norm_obs=True, norm_reward=False)

# Set up logging
log_path = "./a2c_bipedalwalker_logs"
logger = configure(log_path, ["stdout", "csv", "tensorboard"])

# Policy kwargs
policy_kwargs = dict(log_std_init=-2, ortho_init=False)

# Initialize the model
model = A2C('MlpPolicy', env,
            gamma=0.99,
            n_steps=8,
            vf_coef=0.4,
            ent_coef=0.0,
            max_grad_norm=0.5,
            learning_rate=lambda x: x * 0.00096,
            gae_lambda=0.9,
            use_rms_prop=True,
            use_sde=True,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=log_path,
            _init_setup_model=True,
            seed=None)

# Callbacks for checkpointing and evaluations
checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=log_path,
                                          name_prefix='a2c_model')
eval_callback = EvalCallback(env, best_model_save_path=log_path,
                             log_path=log_path, eval_freq=10000,
                             deterministic=True, render=False)

# Learn the model
model.set_logger(logger)
model.learn(total_timesteps=5000000, callback=[checkpoint_callback, eval_callback])

# Save the model
model.save("a2c_bipedalwalker_final")

# Optionally evaluate the model
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"Mean Reward: {mean_reward}, Std Reward: {std_reward}")

# Close the environment
env.close()
