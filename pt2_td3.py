import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure

# Create the environment
env = gym.make('BipedalWalker-v3')

# Specify the policy keyword arguments
policy_kwargs = dict(net_arch=[400, 300])

# Get the shape of the action space
action_dim = env.action_space.shape[0]

# Create action noise with the correct shape
action_noise = NormalActionNoise(mean=np.zeros(action_dim), sigma=0.1 * np.ones(action_dim))

# Setup log directory and configure TensorBoard
log_dir = "logs/TD3_BipedalWalker"
os.makedirs(log_dir, exist_ok=True)
logger = configure(log_dir, ["stdout", "tensorboard"])

# Initialize the TD3 model with the specified hyperparameters
model = TD3('MlpPolicy', env, 
            buffer_size=200000, 
            gamma=0.98, 
            gradient_steps=-1,  # -1 means to perform gradient steps after each environment step
            learning_rate=0.001, 
            learning_starts=10000, 
            train_freq=(1, 'episode'), 
            action_noise=action_noise,
            policy_kwargs=policy_kwargs, 
            verbose=1,
            tensorboard_log=log_dir)

# Set the logger for the model
model.set_logger(logger)

# Train the model
model.learn(total_timesteps=int(1e6))

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f'Mean Reward: {mean_reward}, Std Reward: {std_reward}')

# Save the model for later use
model.save("TD3_BipedalWalker")

# Close the environment
env.close()

# output: Mean Reward: 316.28757343951145, Std Reward: 0.5227015931487111