import json
import matplotlib.pyplot as plt
import numpy as np

# Load episode log
with open('results/evaluation/PE_High_E_Q_episode_rewards_20241208_125059_1a8a5fa4.json', 'r') as file:
    episode_rewards = json.load(file)

with open('results/evaluation/PE_High_E_Q_step_scores_20241208_125059_1a8a5fa4.json', 'r') as file:
    step_scores = json.load(file)

print(len(episode_rewards))
# print(len(episode_rewards))

# Separate episode lengths and total rewards
episode_lengths, total_rewards = zip(*episode_rewards)

# Define a smoothing function using a moving average
def smooth(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Apply smoothing to total rewards and episode lengths
# smoothed_rewards = smooth(total_rewards, window_size=1)
# smoothed_lengths = smooth(episode_lengths, window_size=1)
smoothed_rewards = total_rewards
smoothed_lengths = episode_lengths


# Plot smoothed total rewards
plt.figure(figsize=(10, 6))
plt.plot(range(len(smoothed_rewards)), smoothed_rewards, label="Smoothed Total Rewards")
plt.xlabel("Episode")
plt.ylabel("Total Rewards")
plt.title("Smoothed Total Rewards per Episode")
plt.legend()
plt.grid()
plt.show()

# Plot smoothed episode lengths
plt.figure(figsize=(10, 6))
plt.plot(range(len(smoothed_lengths)), smoothed_lengths, label="Smoothed Episode Lengths", color="orange")
plt.xlabel("Episode")
plt.ylabel("Episode Length (Steps)")
plt.title("Smoothed Episode Lengths")
plt.legend()
plt.grid()
plt.show()

# Plot histogram of rewards
plt.figure(figsize=(10, 6))
plt.hist(total_rewards, bins=15, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel("Total Rewards")
plt.ylabel("Frequency")
plt.title("Distribution of Total Rewards")
plt.grid(axis='y')
plt.show()

# Plot histogram of episode lengths
plt.figure(figsize=(10, 6))
plt.hist(episode_lengths, bins=15, alpha=0.7, color='orange', edgecolor='black')
plt.xlabel("Episode Lengths")
plt.ylabel("Frequency")
plt.title("Distribution of Episode Lengths")
plt.grid(axis='y')
plt.show()


all_step_scores = step_scores

# Compute the average score at each step index

max_length = max(len(steps) for steps in all_step_scores)
average_scores_per_step = []

for i in range(max_length):
    step_values = [steps[i] for steps in all_step_scores if i < len(steps)]
    average_scores_per_step.append(np.mean(step_values))

# Compute cumulative rewards at each step
cumulative_rewards = [sum(steps) for steps in all_step_scores]
smoothed_cumulative_rewards = smooth(cumulative_rewards, window_size=300)

# Plot average scores per step index
plt.figure(figsize=(10, 5))
plt.plot(range(len(average_scores_per_step)), average_scores_per_step, label="Avg Score per Step Index")
plt.xlabel("Step Index")
plt.ylabel("Average Score")
plt.title("Average Scores Per Step Index")
plt.legend()
plt.show()
