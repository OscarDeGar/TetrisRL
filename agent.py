import multiprocessing
import queue
import time

from tetris import Main

def run_tetris(input_queue, output_queue, game_id):
    """Run a single Tetris game instance."""
    print(f"Starting Tetris game {game_id}")
    main = Main(input_queue, output_queue)
    main.run()


if __name__ == "__main__":
    num_games = 3  # Number of Tetris games to run
    input_queues = []
    output_queues = []
    processes = []

    # Set up queues and processes
    for game_id in range(num_games):
        input_queue = multiprocessing.Queue()
        output_queue = multiprocessing.Queue()
        input_queues.append(input_queue)
        output_queues.append(output_queue)

        # Create a process for each Tetris game
        process = multiprocessing.Process(
            target=run_tetris,
            args=(input_queue, output_queue, game_id,)
        )
        processes.append(process)

    # Start all processes
    for process in processes:
        process.start()

    try:
        while True:
            # Loop over each game to send input and receive state
            for game_id in range(num_games):
                try:
                    # Get the current state of the game
                    # print(output_queues[game_id])
                    state = output_queues[game_id].get_nowait()
                    # print(f"Game {game_id} state: {state}")
                except queue.Empty:
                    pass  # No state update available

                # Send random or controlled input to the game
                # Replace this with actual input logic (e.g., from an agent or user)
                # input_action = choice(["left", "right", "rotate", "drop"])
                # input_queues[game_id].put(input_action)

            # Add a short delay to control the update rate
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Shutting down games...")
        for process in processes:
            process.terminate()
        for process in processes:
            process.join()

