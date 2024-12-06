import random
import numpy as np
from collections import defaultdict
from settings import *
from queue import Queue, Empty
import time
import threading
from tetris import Main
import json
import hashlib
import os
import uuid

def default_int():
    return 0

def default_level_3():
    return defaultdict(default_int)

def default_level_2():
    return defaultdict(default_level_3)

def default_level_1():
    return defaultdict(default_level_2)

class SARSA(object):
    def __init__(self, alpha, epsilon, gamma, timeout, Q = None):
        
        ###  HYPERPARAMS
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        ### STATE SPACE
        self.state_space = [[0 for x in range(10)] for y in range(22)]

        ### ACTION SPACE
        self.action_space = {}
        shapes = ['I', 'O', 'T', 'S', 'Z', 'J', 'L']
        orientations = {
            'I': 2,
            'O': 1,
            'T': 4,
            'S': 2,
            'Z': 2,
            'J': 4,
            'L': 4,
        }
        for shape in shapes:
            self.action_space[shape] = {}  # Initialize shape key
            num_orientations = orientations[shape]
            for orientation in range(num_orientations):
                self.action_space[shape][orientation] = []  # Initialize orientation key
                for x in range(10):  # Iterate over positions (0 to 9)
                    self.action_space[shape][orientation].append(x) 

        ### Q - TABLE
        if Q is None:
            self.Q = {}
        else:
            self.Q = Q

        ### TIMEOUT
        self.timeout = timeout

        self.t = 0

        self.episode_lengths = 0
        self.episodes_rewards = []

    def run(self, input_queue, output_queue, game):
        """
        Run the SARSA agent in tetris env(game)

        Args:
            input_queue(Queue): Queue to send actions
            output_queue(Queue): Queue to receive state from game
            game(Tetrs:Main): Tetris agent env
        """
        
        def agent_logic():
            """
            Wrapper function for threading of agent run logic
            """
            ### PARAMS
            state = None
            field_prev = None
            state_prev = None
            action_prev = None
            shape_prev = None
            score_prev = None
            episode_length = 0  # Tracks the length of the current episode
            episode_rewards = []  # Tracks total rewards per episode
            step_scores = []  # Tracks scores at each time step
            all_step_scores = []
            
            # Start the timer
            start_time = time.time()

            ### MAIN LOOP
            while True:
                try:
                    if self.t > self.timeout:
                        elapsed_time = time.time() - start_time
                        print(f"Total experiment duration: {elapsed_time:.2f} seconds", flush=True)

                        # Generate a timestamp for filenames
                        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                        
                        # Ensure the results folder exists
                        results_folder = "results"
                        os.makedirs(results_folder, exist_ok=True)

                        unique_id = uuid.uuid4().hex[:8]  # Short unique identifier

                        # Construct unique filenames
                        q_values_filename = f"q_values_{timestamp}_{unique_id}.json"
                        episode_log_filename = f"episode_log_{timestamp}_{unique_id}.json"

                        # Save Q-values
                        with open(os.path.join(results_folder, q_values_filename), 'w') as file:
                            json.dump(self.Q, file)
                        # print(f"Q-values saved successfully in {q_values_filename}!", flush=True)

                        # Save episode log
                        with open(os.path.join(results_folder, episode_log_filename), 'w') as file:
                            json.dump({
                                "episode_rewards": episode_rewards,
                                "all_step_scores": all_step_scores
                            }, file)
                        # print(f"Episode log saved successfully in {episode_log_filename}!", flush=True)

                        # Save results and quit
                        # self.timeout_save_results(input_queue=input_queue)

                        # Signal the game to quit
                        input_queue.put(["QUIT"])
                        break

                    # GET NEW OUTPUT QUEUE
                    if not output_queue.empty():
                        data = output_queue.get_nowait()  # Get data from the environment
                        field = data["combined_field"]  # Includes both placed blocks and moving tetromino
                        binary_field = data["field_data"]        # Placed blocks only
                        shape = data["shape"]                    # Current tetromino shape
                        score = data["score"]
                        lines = data["lines"]
                        reset = data["reset"]  # Check if the game has reset
                        
                        if reset:
                            # Compute and store episode rewards
                            if episode_length > 0:
                                total_reward = sum(step_scores)
                                episode_rewards.append((episode_length, total_reward))

                                # Save step scores for this episode
                                all_step_scores.append(step_scores.copy())
                                # print(f"Episode ended. Length: {episode_length}, Total Reward: {total_reward}", flush=True)

                            # Reset tracking variables for the new episode
                            state = None
                            field_prev = None
                            state_prev = None
                            action_prev = None
                            shape_prev = None
                            score_prev = None
                            episode_length = 0
                            step_scores = []  # Reset here!
                            continue
                                                
                        # CHECK FOR FIELD CHANGES
                        combined_changed = field_prev is None or any(
                            row1 != row2 for row1, row2 in zip(field_prev, field)
                        )

                        if combined_changed:
                            # STATE CLEAN-UP
                            state = tuple(tuple(row) for row in binary_field)

                            # ACTION SELECTION
                            # If there's a previous state and action, update Q-value
                            if state_prev is not None and action_prev is not None:
                                print(f"TIME: {self.t}", flush=True)
                                # Calculate reward
                                reward = self.calculate_reward(state_prev, state, score_prev, score, lines)
                                # print(f"Reward: {reward}", flush=True)

                                # Score update
                                step_scores.append(reward)
                                
                                # Select the next action
                                next_action = self.choose_action(state, shape)
                                # print(f"ACTION: {next_action}",flush=True)
                                # Update Q-value
                                self.update_q_value(state_prev, action_prev, shape_prev, reward, state, next_action, shape)

                                # Debugging
                                # print(f"Updated Q-value for {state_prev}, {action_prev}: {self.Q[state_prev][shape][action_prev[0]][action_prev[1]]}", flush=True)

                                # Move to the next state-action pair
                                state_prev = state
                                action_prev = next_action
                                shape_prev = shape
                                score_prev = score
                                self.t += 1  # Increment time step or episode count
                                episode_length += 1  # Increment the current episode length

                            # Select an action if no previous action exists
                            if action_prev is None:
                                print(f"TIME: {self.t}", flush=True)
                                action_prev = self.choose_action(state, shape)
                                state_prev = state
                                shape_prev = shape
                                score_prev = score
                                self.t += 1  # Increment time step or episode count
                                episode_length += 1  # Increment the current episode length

                            # Perform the selected action
                            commands = self.convert_action(action_prev)
                            input_queue.put(commands)

                        # Update combined_prev
                        field_prev = [row.copy() for row in field]

                except Empty:
                    time.sleep(0.1)  # Sleep briefly to avoid busy-waiting
                except Exception as e:
                    print(f"Agent encountered an error: {e}")
                    break
        
        # START AGENT THREAD
        agent_thread = threading.Thread(target=agent_logic, daemon=True)
        agent_thread.start()

        # RUN TETRIS AGENT
        game.run()

    def choose_action(self, state, shape):
        """
        Choose an action using epsilon-greedy policy

        Args:
            state(self.state): current windowed state
            shape(str): letter name of selected piece
        Return:
            action: selected action as (orientation, pos.x)
        """
        state_key = self.hash_state(state)

        ### RANDOM GREEDY(EXPLORE)
        if np.random.rand() < self.epsilon:
            # Explore: randomly select an orientation and position
            orientation = random.choice(list(self.action_space[shape].keys()))
            pos_x = random.choice(self.action_space[shape][orientation])
            return (orientation, pos_x)
        ### EXPLOIT
        else:

            max_q = float('-inf')
            best_action = None
            # SWEEP (orientation,poses) FOR state
            for orientation in self.action_space[shape].keys():
                # print(orientation, flush=True)
                for pos_x in self.action_space[shape][orientation]:
                    # print(pos_x, flush=True)

                    # Q-VALUE LOOKUP
                    q_value = self.get_q_value(state_key, shape, orientation, pos_x)
                    # print(q_value,flush=True)
                    # print(f"Q-value for ({state}, {shape}, {orientation}, {pos_x}): {q_value}", flush=True)

                    # MAX VALUE CHECK
                    if q_value > max_q:
                        max_q = q_value
                        best_action = (orientation, pos_x)

            # print("Exploit Done", flush=True)

            # IF NO MAX_Q, RANDOM ACTION(DEFAULT)
            if max_q == 0:
                orientation = random.choice(list(self.action_space[shape].keys()))
                pos_x = random.choice(self.action_space[shape][orientation])
                return (orientation, pos_x)
            # RETURN ACTION
            return best_action

    def update_q_value(self, state, action, shape_prev, reward, next_state, next_action, shape):
        """
        Update the Q-value using the SARSA update rule.
        """
        # Abstract the current and next states
        state_key = self.hash_state(state)
        next_state_key = self.hash_state(next_state)

        # Retrieve current Q-value
        current_q = self.get_q_value(state_key, shape_prev, action[0], action[1])
        # self.Q[state_key][shape_prev][action[0]].get(action[1], 0)
        print(f"CURR_q  {current_q}", flush=True)

        # Retrieve next Q-value
        next_q = self.get_q_value(next_state_key, shape, next_action[0], next_action[1]) 
        # self.Q[next_state_key][shape][next_action[0]].get(next_action[1], 0)

        # SARSA update
        self.Q[state_key][shape_prev][action[0]][action[1]] += self.alpha * (reward + self.gamma * next_q - current_q)
        print(f"NEW Q: {self.Q[state_key][shape_prev][action[0]][action[1]]}", flush=True)

    def get_q_value(self, state, shape, orientation, pos_x, default=0):
        """
        Safely retrieve the Q-value for the given state, shape, orientation, and pos_x.
        Ensures all intermediate keys are handled correctly.
        """
        if state not in self.Q:
            self.Q[state] = {}
        if shape not in self.Q[state]:
            self.Q[state][shape] = {}
        if orientation not in self.Q[state][shape]:
            self.Q[state][shape][orientation] = {}
        if pos_x not in self.Q[state][shape][orientation]:
            self.Q[state][shape][orientation][pos_x] = default
        return self.Q[state][shape][orientation][pos_x]
    
    def calculate_reward(self, prev_state, current_state, score_prev, score, lines):
        """
        Calculate reward based on the transition between states.
        
        Args:
            prev_state: The previous game state.
            current_state: The current game state.

        Returns:
            int: The computed reward.
        """
        # Example reward: Count completed rows
        score_diff = score - score_prev

        line_factor = lines * 0.1

        fill_factor = self.filled_factor(current_state)
        height_scaler = self.height_factor(current_state,"Pos")

        fill_r = fill_factor + (fill_factor * height_scaler)
        
        pos_r = score_diff + (score_diff*line_factor) + fill_r  

        height_pen = self.height_factor(current_state,"Neg")
        hole_pen = self.hole_factor(current_state)
        # print(height_pen,flush=True)
        # print(hole_pen,flush=True)
        # neg_r =  hole_pen + (hole_pen * height_pen)
        neg_r = hole_pen

        reward = pos_r - neg_r

        return reward
        
    def convert_action(self,  action): 
        commands = []

        orientation, pos_x = action
        orientation_diff = (orientation - 0) % 4
        commands.extend(['K_LCTRL'] * orientation_diff)

        pos_diff = pos_x - 5
        if pos_diff > 0:
            # Move right
            commands.extend(['K_RIGHT'] * pos_diff)
        elif pos_diff < 0:
            # Move left
            commands.extend(['K_LEFT'] * abs(pos_diff))

        commands.extend(['K_DOWN'])
        return commands
    
    def filled_factor(self, state):
        """
        Calculate the filled factor for the state.

        For each row in the state, count the number of 1's and multiply the count by a scalar
        based on the row index mapping from 2-22 to 0-0.15. Sum the contributions for all rows.

        Args:
            state: The game state as a 2D list.

        Returns:
            float: The computed filled factor (single scalar value for the board).
        """
        def map_row_index_to_scale(index):
            """Map row index (2-22) to a scalar between 0 and 0.15."""
            min_index, max_index = 2, 22
            min_scale, max_scale = 0, 0.8
            return min_scale + (max_scale - min_scale) * (index - min_index) / (max_index - min_index)
    
        total_factor = 0.0

        for index, row in enumerate(state):
            if index < 2 or index > 22:
                continue  # Skip rows outside the 2-22 range
            count = sum(row)  # Count the number of 1's in the row
            scale = map_row_index_to_scale(index)  # Map the index to the scale
            total_factor += count * scale  # Multiply the count by the scale and add to the total

        return total_factor
    
    def height_factor(self, state, use):
        """
        Calculate the maximum height of the occupied blocks.
        Higher blocks correspond to higher height values.
        
        Args:
            field: The game field (list of lists)
        
        Returns:
            int: The height of the highest occupied block
        """
        ind_high = 2
        ind_low = 22
        scale_high_pos = 0
        scale_low_pos = 0.5
        scale_high_neg = 0.25
        scale_low_neg = 0  

        for row_index, row in enumerate(state):
            if any(cell == 1 for cell in row):
                if use == "Pos":
                    return abs(scale_low_pos + (scale_high_pos - scale_low_pos) * (row_index - ind_low) / (ind_high - ind_low))
                else:
                    return abs(scale_low_neg + (scale_high_neg - scale_low_neg) * (row_index - ind_low) / (ind_high - ind_low))
                # return rows - row_index  # Height from the bottom
        return 0  # No blocks, height is 0
    
    def hole_factor(self, state):
        """
        Calculate the number of holes in the state.
        A hole is defined as a '0' below the first '1' encountered in each column.
        If all cells below the first '1' are also '1', no holes are counted for that column.

        Args:
            state: A 2D list representing the current game state.

        Returns:
            int: The total number of holes in the state.
        """
        holes = 0
        columns = len(state[0])
        rows = len(state)
        
        for col in range(columns):
            block_found = False
            column_holes = 0
            valid_column = True
            for row in range(rows):
                cell = state[row][col]
                if cell == 1:
                    block_found = True
                elif cell == 0 and block_found:
                    column_holes += 1
                elif cell == 1 and block_found:
                    valid_column = False  # The column has non-hole blocks below the first '1'

            if column_holes > 0 and valid_column:
                holes += column_holes

        return holes
    
    def hash_state(self, state):
        """
        Convert a binary state representation into a hashed key for compact storage.

        Args:
            state: A list of lists representing the binary state (e.g., 22 rows of 10 binary values).

        Returns:
            str: A hashed string representing the state.
        """
        # Convert each row into its decimal representation
        row_values = [int("".join(map(str, row)), 2) for row in state]
        
        # Join the row values with a delimiter
        state_string = "-".join(map(str, row_values))
        
        # Hash the resulting string to create a compact key
        hashed_key = hashlib.sha256(state_string.encode()).hexdigest()
        
        return hashed_key
   

if __name__ == "__main__":
    # Create input and output queues
    input_queue = Queue()
    output_queue = Queue()

    # Initialize the game
    tetris = Main(input_queue=input_queue, output_queue=output_queue)

    # Load previous Q
    with open('results/q_values_20241205_181140_795a4f86.json', 'r') as file:
        policy = json.load(file)
    # Initialize the SARSA agent
    sarsa_agent = SARSA(alpha=0.15, epsilon=0.05, gamma=0.90, timeout=200000, Q = policy)

    # Run the SARSA agent
    sarsa_agent.run(input_queue, output_queue, tetris)
