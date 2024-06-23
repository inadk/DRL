import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from scipy.stats import sem

# Define seeds and masses used for training and testing
seeds = [1, 2, 3, 4, 5]
training_masses = [3, 6, 9]
test_masses = [3, 4, 5, 6, 7, 8, 9]

class ChangeMassWrapper(gym.Wrapper):
    def __init__(self, env, torso_mass=6):
        super().__init__(env)
        self.torso_mass = torso_mass
        self.update_mass()

    def update_mass(self):
        self.env.model.body_mass[1] = self.torso_mass

    def set_mass(self, new_mass):
        self.torso_mass = new_mass
        self.update_mass()

def evaluate_model(model, env, num_episodes=10):
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=num_episodes, return_episode_rewards=False)
    return mean_reward, std_reward

def test_models(seeds, masses, test_masses):
    results = {}
    for seed in seeds:
        for mass in masses:
            model_path = f"ppo_hopper_seed_{seed}_mass_{mass}_hopperv3params.zip"
            env_stats_path = f"ppo_hopper_vec_normalize_seed_{seed}_mass_{mass}_hopperv3params.pkl"

            env = make_vec_env('Hopper-v4', n_envs=1, seed=seed, wrapper_class=ChangeMassWrapper, wrapper_kwargs={'torso_mass': mass})
            env = VecNormalize.load(env_stats_path, env)
            model = PPO.load(model_path, env=env)

            model_results = []
            for test_mass in test_masses:
                env.env_method("set_mass", test_mass)
                mean_reward, std_reward = evaluate_model(model, env)
                model_results.append((mean_reward, std_reward))
                print(f"Seed {seed}, Train Mass {mass}, Test Mass {test_mass}: Mean Reward = {mean_reward}, Std Reward = {std_reward}")

            results[(seed, mass)] = model_results

    return results

# Test the models
results = test_models(seeds, training_masses, test_masses)

# Save the results
np.save("ppo_hopper_results.npy", results)

def plot_results(results, test_masses, training_masses):
    # Determine the number of unique training masses and assign colors
    unique_masses = sorted(set(mass for _, mass in results.keys()))
    colors = ['blue', 'green', 'red']  # Colors for each training mass
    legend_positions = ['upper right', 'upper right', 'upper left']  # Custom positions for legends
    fig, axes = plt.subplots(nrows=1, ncols=len(unique_masses), figsize=(7.5, 3))  # Adjusted width and height
    
    # Plot each set of results on a different subplot
    for ax, mass, color, legend_pos in zip(axes, unique_masses, colors, legend_positions):
        mass_results = {k: v for k, v in results.items() if k[1] == mass}
        
        # Initialize lists to store the aggregated data for plotting
        mean_values = []
        upper_bounds = []
        lower_bounds = []
        
        # Calculate mean and confidence interval for each test mass
        for test_mass in test_masses:
            all_rewards = [rewards[test_masses.index(test_mass)][0] for _, rewards in mass_results.items()]
            all_stds = [rewards[test_masses.index(test_mass)][1] for _, rewards in mass_results.items()]
            mean_rewards = np.mean(all_rewards)
            mean_std = np.mean(all_stds)
            
            # Collect the data
            mean_values.append(mean_rewards)
            upper_bounds.append(mean_rewards + mean_std)
            lower_bounds.append(mean_rewards - mean_std)
        
        # Plotting the mean, upper, and lower bounds
        test_mass_indices = np.arange(len(test_masses))
        ax.plot(test_mass_indices, mean_values, color=color, linewidth=2)  # Thicker line for mean
        ax.plot(test_mass_indices, upper_bounds, color=color, linestyle='--')
        ax.plot(test_mass_indices, lower_bounds, color=color, linestyle='--')
        
        # Filling the area between the confidence bounds
        ax.fill_between(test_mass_indices, lower_bounds, upper_bounds, color=color, alpha=0.3)
        
        ax.set_xticks(test_mass_indices)
        ax.set_xticklabels(test_masses)
        ax.set_xlabel('Torso Mass')
        
        # Only set the y-label on the first subplot
        if ax == axes[0]:
            ax.set_ylabel('Performance')
        
        ax.grid(True)
        
        # Add legend to each subplot with custom positions
        ax.legend([f'm={mass}'], loc=legend_pos)
    
    plt.tight_layout(pad=0.4)  # Tight layout with minimal padding
    plt.show()

    plt.savefig("ppo_hopper_results1.png")

# Plot the results
plot_results(results, test_masses, training_masses)
