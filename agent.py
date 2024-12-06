from queue import Queue, Empty
import time
import threading

from tetris import Main

# Start a thread or process for the agent
def agent_logic(input_queue, output_queue):
    while True:
        try:
            # Retrieve the game state
            if not output_queue.empty():
                state = output_queue.get()
                # print("Received state:")
                print(state)
                # for row in state["field_data"]:
                #     print(row)

                # Generate an action based on the state (example action)
                action = {"action": "move_left"}
                input_queue.put(action)
        except Empty:
            # Queue is empty; sleep briefly to avoid busy-waiting
            time.sleep(0.01)
        except Exception as e:
            print(f"Agent encountered an error: {e}")
            break


if __name__ == "__main__":

    # Create input and output queues
    input_queue = Queue()
    output_queue = Queue()

    # Initialize the game
    tetris = Main(input_queue=input_queue, output_queue=output_queue)

    agent_thread = threading.Thread(target=agent_logic, args=(input_queue, output_queue), daemon=True)
    agent_thread.start()

    # Run the game
    tetris.run()


