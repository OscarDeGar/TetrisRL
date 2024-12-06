

# ### MAIN LOOP
#             while True:
#                 try:
#                     # GET NEW OUTPUT QUEUE
#                     if not output_queue.empty():
#                         data = output_queue.get_nowait()  # Get data from the environment
#                         field = data["combined_field"]
#                         binary_field = data["field_data"]  # Extract the field data

#                         # CHECK FOR FIELD CHANGES
#                         if field_prev == None or any(row1 != row2 for row1, row2 in zip(field_prev, field)):
#                             # print("Environment state changed, processing action.")
#                             for ind, row in enumerate(binary_field[3:]):
#                                 if any(item == 1 for item in row):  # Check for blocks in the row
#                                     if ind < 7:
#                                         state = binary_field[2:9]  # Fixed window if near the top
#                                         break
#                                     elif ind < 15:
#                                         state = binary_field[ind:ind + 7]  # Rolling window for middle rows
#                                         break
#                                     else:
#                                         state = binary_field[15:]  # Fixed window for near the bottom
#                                         break  # Stop after processing the first relevant row
#                                 else:
#                                     state = binary_field[15:]

#                             # STATE CLEAN-UP
#                             state = tuple(tuple(row) for row in state)

#                             ### DEBUG
#                             # for row in state:
#                             #     print(row,flush=True)
#                             # print("prev",flush=True)
#                             # if state_prev is not None:
#                             #     for row in state_prev:
#                             #         print(row,flush=True)
#                             # print(data["shape"],flush=True)
#                             # print(state,flush=True)
#                             # print("curr",flush=True)
#                             # for row in binary_field:
#                             #     print(row,flush=True)
#                             # print("prev",flush=True)
#                             # if binary_prev is not None:
#                             #     for row in binary_prev:
#                             #         print(row,flush=True)

#                             # ACTION SELECTION
#                             if state_prev != state or not action_pending:
#                                 action = self.choose_action(state, data["shape"])
#                                 print(f"Chosen Action: {action}", flush=True)

#                                 # Update binary_prev BEFORE setting action_pending
#                                 binary_prev = [row.copy() for row in binary_field]

#                                 # # Send the action to the input queue
#                                 # input_queue.put({"action": action})

#                                 # Set action_pending to True after sending the action
#                                 action_pending = True

#                                 # Update state_prev
#                                 state_prev = state
#                                 for row in state:
#                                     print(row,flush=True)
#                             else:
#                                 binary_prev = [row.copy() for row in binary_field]

#                                 # Find the lowest row with a change in the binary_field
#                                 for changed_ind, (prev_row, curr_row) in enumerate(zip(binary_prev, binary_field)):
#                                     if any(item1 != item2 for item1, item2 in zip(prev_row, curr_row)):
#                                         # Recalculate the rolling window based on the changed index
#                                         if changed_ind < 7:
#                                             state = binary_field[2:9]  # Fixed window if near the top
#                                             break
#                                         elif changed_ind < 15:
#                                             state = binary_field[changed_ind:changed_ind + 7]  # Rolling window for middle rows
#                                             break
#                                         else:
#                                             state = binary_field[15:]  # Fixed window for near the bottom
#                                             break
#                                     else:
#                                         state = binary_field[15:]

#                                 # Convert to tuple for consistent state representation
#                                 state = tuple(tuple(row) for row in state)

#                                 # Select a new action based on the updated state
#                                 action = self.choose_action(state, data["shape"])
#                                 print(f"Edge Case Chosen Action: {action}", flush=True)

#                                 # # Send the action to the input queue
#                                 # input_queue.put({"action": action})

#                                 # Set action_pending to True after sending the action
#                                 action_pending = True

#                                 # Update state_prev
#                                 state_prev = state
#                                 for row in state:
#                                     print(row,flush=True)

#                             # UPDATE PREV VARS
#                             field_prev = [row.copy() for row in field]
#                             # state_prev = state