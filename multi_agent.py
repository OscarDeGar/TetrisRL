from multiprocessing import Process, Queue
import pickle
from collections import defaultdict
from tetris import Main
import json
import copy
import queue

from sarsa import SARSA

def worker(worker_id, input_queue, output_queue, policy, alpha, epsilon, gamma, timeout, result_queue):
    """
    Worker function to run a single Tetris game and SARSA agent.
    """
    print(f"Worker {worker_id} started.", flush=True)
    
    # Initialize the game
    tetris = Main(input_queue=input_queue, output_queue=output_queue)
    
    # Initialize the SARSA agent with the given policy
    sarsa_agent = SARSA(alpha=alpha, epsilon=epsilon, gamma=gamma, timeout=timeout, Q=policy)
    
    # Run the SARSA agent
    sarsa_agent.run(input_queue, output_queue, tetris)
    # print("AGENTDONE",flush=True)
    # Save the resulting policy and metrics
    worker_policy = sarsa_agent.Q

    # if worker_policy:
    #     print("POLICYS",flush=True)
    
    print(f"Worker {worker_id} finished. Sending results.", flush=True)
    # print(worker_policy,flush=True)
    # Send results to the main process
    result_queue.put_nowait({
        "worker_id": worker_id,
        "policy": worker_policy,
    })


def main(num_workers, alpha, epsilon, gamma, timeout, num_generations):
    """
    Main function to manage multiple workers and evolve policies over generations.
    """
    result_queue = Queue()
    policies = [None] * num_workers

    for generation in range(num_generations):
        print(f"Starting generation {generation}...", flush=True)
        processes = []

        for worker_id in range(num_workers):
            input_queue = Queue()
            output_queue = Queue()
            
            p = Process(
                target=worker,
                args=(worker_id, input_queue, output_queue, policies[worker_id], alpha, epsilon, gamma, timeout, result_queue)
            )
            processes.append(p)
            p.start()

        # Collect results from workers before joinin

        # Now join the processes
        for p in processes:
            p.join()

        # input_queue.close()
        # input_queue.join_thread()
        # output_queue.close()
        # output_queue.join_thread()

        results = []
        for _ in range(num_workers):
            try:
                result = result_queue.get(timeout=5)
                results.append(result)
            except queue.Empty:
                print(f"Result queue timeout for one of the workers in generation {generation}.", flush=True)

        if not results:
            print("No results collected. Exiting.", flush=True)
            break

        policies = evolve_policies(results)
        print(f"Generation {generation} policies updated.", flush=True)

    final_policy = merge_policies(policies)
    save_final_policy(final_policy, "final_policy.json")
    print("Final merged policy saved successfully!", flush=True)


def merge_policies(policies):
    """
    Merge multiple worker policies into a single policy without using defaultdict or lambda functions.

    Args:
        policies (list): List of Q-tables (dicts) from all workers.

    Returns:
        dict: A merged policy.
    """
    merged_policy = {}
    num_workers = len(policies)

    for policy in policies:
        for state, shape_dict in policy.items():
            if state not in merged_policy:
                merged_policy[state] = {}
            for shape, orientation_dict in shape_dict.items():
                if shape not in merged_policy[state]:
                    merged_policy[state][shape] = {}
                for orientation, position_dict in orientation_dict.items():
                    if orientation not in merged_policy[state][shape]:
                        merged_policy[state][shape][orientation] = {}
                    for pos_x, q_value in position_dict.items():
                        if pos_x not in merged_policy[state][shape][orientation]:
                            merged_policy[state][shape][orientation][pos_x] = 0.0
                        # Add the Q-values for merging
                        merged_policy[state][shape][orientation][pos_x] += q_value

    # Average Q-values if the same state-action pair exists in multiple policies
    for state in merged_policy:
        for shape in merged_policy[state]:
            for orientation in merged_policy[state][shape]:
                for pos_x in merged_policy[state][shape][orientation]:
                    merged_policy[state][shape][orientation][pos_x] /= num_workers

    return merged_policy


def save_final_policy(policy, filename):
    """
    Save the final merged policy to a JSON file without using defaultdict or lambda functions.

    Args:
        policy (dict): The final merged policy.
        filename (str): The file name for saving the policy.
    """
    # Ensure all keys are converted to strings for JSON serialization
    def convert_keys_to_strings(obj):
        if isinstance(obj, dict):
            return {str(k): convert_keys_to_strings(v) for k, v in obj.items()}
        else:
            return obj  # base case, not a dict

    serializable_policy = convert_keys_to_strings(policy)

    with open(filename, 'w') as file:
        json.dump(serializable_policy, file)

def evolve_policies(results):
    """
    Evolve policies based on results from all workers without using defaultdict or lambda functions.

    Args:
        results (list): List of dictionaries containing worker results, each with a "policy" key.

    Returns:
        list: A list of evolved policies, one for each worker.
    """
    policies = [result["policy"] for result in results]

    # Flatten policies into a single-level dictionary
    flat_combined_policy = {}
    policy_counts = {}
    print("here", flush=True)

    for policy in policies:
        for state, shapes in policy.items():
            for shape, orientations in shapes.items():
                for orientation, positions in orientations.items():
                    for pos_x, value in positions.items():
                        key = (state, shape, orientation, pos_x)
                        if key not in flat_combined_policy:
                            flat_combined_policy[key] = 0.0
                            policy_counts[key] = 0
                        flat_combined_policy[key] += value
                        policy_counts[key] += 1
    print("policy checked", flush=True)

    # Average the flattened values
    for key in flat_combined_policy:
        flat_combined_policy[key] /= policy_counts[key]

    # Reconstruct the combined policy
    combined_policy = {}
    for (state, shape, orientation, pos_x), value in flat_combined_policy.items():
        if state not in combined_policy:
            combined_policy[state] = {}
        if shape not in combined_policy[state]:
            combined_policy[state][shape] = {}
        if orientation not in combined_policy[state][shape]:
            combined_policy[state][shape][orientation] = {}
        if pos_x not in combined_policy[state][shape][orientation]:
            combined_policy[state][shape][orientation][pos_x] = 0.0
        combined_policy[state][shape][orientation][pos_x] = value

    # Clone the combined policy for all workers
    new_policies = [copy.deepcopy(combined_policy) for _ in policies]
    return new_policies

if __name__ == "__main__":
    num_workers = 5  # Number of parallel Tetris games
    alpha = 0.15
    epsilon = 0.05
    gamma = 0.90
    timeout = 50  # Number of steps per worker
    num_generations = 3  # Number of policy updates

    main(num_workers, alpha, epsilon, gamma, timeout, num_generations)    
