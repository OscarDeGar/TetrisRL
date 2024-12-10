import random
import json
import hashlib
import time
import threading
from queue import Queue, Empty
import os
import uuid
from tetris import Main

class SarsaEvaluator:
    def __init__(self, policy=None, ep_timeout=1000, file_prefix=None):
        """
        Initialize the PolicyEvaluator.

        Args:
            policy (dict): Predefined policy as a dictionary. If None, starts with an empty policy.
            ep_timeout (int): Maximum number of steps per episode before timeout.
            file_prefix (str): Prefix for saving output files.
        """
        self.Q = policy if policy else {}
        self.ep_timeout = ep_timeout
        self.file_prefix = file_prefix or "PolicyEval"
        self.episode_rewards = []  # Cumulative policy reward per episode
        self.all_step_rewards = []  # Step-wise policy rewards per episode
        self.all_score_differences = []  # Step-wise score differences per episode
        self.ep_t = 0  # Episode counter
        self.in_game_scores = []  # Final in-game score for each episode

        # Define action space
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
            self.action_space[shape] = {}
            num_orientations = orientations[shape]
            for orientation in range(num_orientations):
                self.action_space[shape][orientation] = list(range(10))  # Positions 0 to 9

    def run(self, input_queue, output_queue, game):
        """
        Run the PolicyEvaluator agent in the Tetris environment.

        Args:
            input_queue (Queue): Queue to send actions.
            output_queue (Queue): Queue to receive state from the game.
            game (Main): Tetris game environment.
        """
        def agent_logic():
            state = None
            field_prev = None
            score_prev = 0
            episode_length = 0
            step_policy_rewards = []
            step_score_differences = []
            cumulative_policy_reward = 0
            start_time = time.time()

            while True:
                try:
                    if not output_queue.empty():
                        data = output_queue.get_nowait()
                        field = data["combined_field"]
                        binary_field = data["field_data"]
                        shape = data["shape"]
                        score = data["score"]
                        lines = data["lines"]
                        reset = data["reset"]

                        # Handle game reset (episode ends)
                        if reset:
                            if episode_length > 0:
                                self.episode_rewards.append((episode_length, cumulative_policy_reward))
                                self.all_step_rewards.append(step_policy_rewards.copy())
                                self.all_score_differences.append(step_score_differences.copy())
                                self.in_game_scores.append(score)
                                print(f"Episode {self.ep_t} ended. Length: {episode_length}, "
                                      f"Policy Reward: {cumulative_policy_reward}, Final Score: {score}")

                            # Increment episode counter
                            self.ep_t += 1

                            # Check if maximum episodes reached
                            if self.ep_t >= self.ep_timeout:
                                self.save_metrics(start_time)
                                input_queue.put(["QUIT"])
                                break

                            # Reset variables for the new episode
                            state = None
                            field_prev = None
                            score_prev = 0
                            episode_length = 0
                            step_policy_rewards = []
                            step_score_differences = []
                            cumulative_policy_reward = 0
                            continue

                        # Detect field changes
                        combined_changed = field_prev is None or any(
                            row1 != row2 for row1, row2 in zip(field_prev, field)
                        )

                        if combined_changed:
                            state = tuple(tuple(row) for row in binary_field)

                            # Calculate rewards
                            policy_reward = self.calculate_reward(field, score_prev, score, lines)
                            # print(policy_reward,flush=True)
                            score_diff = score - (score_prev or 0)
                            cumulative_policy_reward += policy_reward
                            step_policy_rewards.append(policy_reward)
                            step_score_differences.append(score_diff)

                            # Choose and execute action
                            action = self.choose_action(state, shape)
                            commands = self.convert_action(action)
                            input_queue.put(commands)

                            # Update previous state and score
                            field_prev = [row.copy() for row in field]
                            score_prev = score
                            episode_length += 1

                except Empty:
                    time.sleep(0.1)
                except Exception as e:
                    print(f"Agent encountered an error: {e}")
                    break

        agent_thread = threading.Thread(target=agent_logic, daemon=True)
        agent_thread.start()
        game.run()

    def choose_action(self, state, shape):
        """
        Choose an action based on the predefined policy or select a random action.

        Args:
            state (tuple): Current state of the environment.
            shape (str): Current shape of the Tetris piece.

        Returns:
            tuple: Selected action as (orientation, pos.x).
        """
        state_key = self.hash_state(state)
        if state_key in self.Q and shape in self.Q[state_key]:
            
            max_q = float('-inf')
            best_action = None

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

                    # IF NO MAX_Q, RANDOM ACTION(DEFAULT)
            if max_q == 0:
                orientation = random.choice(list(self.action_space[shape].keys()))
                pos_x = random.choice(self.action_space[shape][orientation])
                return (orientation, pos_x)
            # RETURN ACTION
            return best_action
        else:
            orientation = random.choice(list(self.action_space[shape].keys()))
            pos_x = random.choice(self.action_space[shape][orientation])
            return (orientation, pos_x)
        
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

    def convert_action(self, action):
        """
        Convert the chosen action into game commands.

        Args:
            action (tuple): Action as (orientation, pos.x).

        Returns:
            list: List of commands to execute the action in the game.
        """
        commands = []
        orientation, pos_x = action
        orientation_diff = (orientation - 0) % 4
        commands.extend(['K_LCTRL'] * orientation_diff)

        pos_diff = pos_x - 5
        if pos_diff > 0:
            commands.extend(['K_RIGHT'] * pos_diff)
        elif pos_diff < 0:
            commands.extend(['K_LEFT'] * abs(pos_diff))

        commands.extend(['K_DOWN'])
        return commands

    def calculate_reward(self, current_state, score_prev, score, lines):
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
        neg_r =  hole_pen + (hole_pen * height_pen)
        # neg_r = hole_pen

        reward = pos_r - neg_r

        return reward
    
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
        def map_row_index_to_scale(index, state):
            """Map row index (2-22) to a scalar between 0 and 0.15."""
            min_index, max_index = 4, len(state)
            min_scale, max_scale = 0, 0.8
            return min_scale + (max_scale - min_scale) * (index - min_index) / (max_index - min_index)
    
        total_factor = 0.0

        for index, row in enumerate(state):
            if index < 4 or index > (len(state)):
                continue  # Skip rows outside the 2-22 range
            count = sum(row)  # Count the number of 1's in the row
            scale = map_row_index_to_scale(index,state)  # Map the index to the scale
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
        ind_high = 4
        ind_low = len(state)
        scale_high_pos = 0
        scale_low_pos = 0.5
        scale_high_neg = 0.1
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
            state: A list of lists representing the binary state.

        Returns:
            str: A hashed string representing the state.
        """
        row_values = [int("".join(map(str, row)), 2) for row in state]
        state_string = "-".join(map(str, row_values))
        hashed_key = hashlib.sha256(state_string.encode()).hexdigest()
        return hashed_key

    def save_metrics(self, start_time):
        """
        Save the recorded metrics to files.

        Args:
            start_time (float): Start time of the evaluation.
        """
        elapsed_time = time.time() - start_time
        print(f"Total evaluation duration: {elapsed_time:.2f} seconds")

        results_folder = "results/evaluation"
        os.makedirs(results_folder, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        unique_id = uuid.uuid4().hex[:8]
        rewards_file = f"{self.file_prefix}_episode_rewards_{timestamp}_{unique_id}.json"
        steps_file = f"{self.file_prefix}_step_scores_{timestamp}_{unique_id}.json"

        with open(os.path.join(results_folder, rewards_file), 'w') as file:
            json.dump(self.episode_rewards, file)
        print(f"Episode rewards saved in {rewards_file}.")

        with open(os.path.join(results_folder, steps_file), 'w') as file:
            json.dump(self.all_step_scores, file)
        print(f"Step scores saved in {steps_file}.")


if __name__ == "__main__":
    # Create input and output queues
    input_queue = Queue()
    output_queue = Queue()

    # Initialize the game
    tetris = Main(input_queue=input_queue, output_queue=output_queue)

    # Load the policy
    with open('results/Test_High_E_q_values_small_20241208_095222_6eb29a27.json', 'r') as file:
        policy = json.load(file)

    # Initialize the PolicyEvaluator
    evaluator = SarsaEvaluator(policy=policy, ep_timeout=1000, file_prefix="PE_High_E_Q")

    # Run the PolicyEvaluator
    evaluator.run(input_queue, output_queue, tetris)
