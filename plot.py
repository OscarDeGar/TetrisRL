import json
import matplotlib.pyplot as plt
import numpy as np

# Load episode log
with open('results/episode_log_20241205_195859_1ef8caf1.json', 'r') as file:
    log = json.load(file)

episode_rewards = log["episode_rewards"]
step_scores = log["all_step_scores"]
# print(len(episode_rewards))

# Separate episode lengths and total rewards
episode_lengths, total_rewards = zip(*episode_rewards)

# Define a smoothing function using a moving average
def smooth(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Apply smoothing to total rewards and episode lengths
smoothed_rewards = smooth(total_rewards, window_size=100)
smoothed_lengths = smooth(episode_lengths, window_size=100)

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
plt.hist(total_rewards, bins=20, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel("Total Rewards")
plt.ylabel("Frequency")
plt.title("Distribution of Total Rewards")
plt.grid(axis='y')
plt.show()

# Plot histogram of episode lengths
plt.figure(figsize=(10, 6))
plt.hist(episode_lengths, bins=20, alpha=0.7, color='orange', edgecolor='black')
plt.xlabel("Episode Lengths")
plt.ylabel("Frequency")
plt.title("Distribution of Episode Lengths")
plt.grid(axis='y')
plt.show()


all_step_scores = log['all_step_scores']

# Compute the average score at each step index

max_length = max(len(steps) for steps in all_step_scores)
average_scores_per_step = []

for i in range(max_length):
    step_values = [steps[i] for steps in all_step_scores if i < len(steps)]
    average_scores_per_step.append(np.mean(step_values))

# Compute cumulative rewards at each step
cumulative_rewards = [sum(steps) for steps in all_step_scores]
smoothed_cumulative_rewards = smooth(cumulative_rewards, window_size=100)

# Plot average scores per step index
plt.figure(figsize=(10, 5))
plt.plot(range(len(average_scores_per_step)), average_scores_per_step, label="Avg Score per Step Index")
plt.xlabel("Step Index")
plt.ylabel("Average Score")
plt.title("Average Scores Per Step Index")
plt.legend()
plt.show()

# Plot cumulative rewards
plt.figure(figsize=(10, 5))
plt.plot(range(len(smoothed_cumulative_rewards)), smoothed_cumulative_rewards, label="Cumulative Reward")
plt.xlabel("Episode Index")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Rewards Across Episodes")
plt.legend()
plt.show()

# # Plot step scores
# plt.figure(figsize=(10, 6))
# plt.plot(range(len(step_scores)), step_scores, label="Step Rewards", color="green")
# plt.xlabel("Time Step")
# plt.ylabel("Step Reward")
# plt.title("Rewards at Each Time Step")
# plt.legend()
# plt.grid()
# plt.show()