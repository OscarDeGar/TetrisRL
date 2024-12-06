from settings import *
from sys import exit
from os.path import join

# components
from game import Game
from score import Score
from preview import Preview

from random import choice

import time

class Main:
	def __init__(self, input_queue=None, output_queue=None):

		# general 
		pygame.init()
		self.display_surface = pygame.display.set_mode((WINDOW_WIDTH,WINDOW_HEIGHT))
		self.clock = pygame.time.Clock()
		pygame.display.set_caption('Tetris')

		# shapes
		self.next_shapes = [choice(list(TETROMINOS.keys())) for shape in range(3)]

		# components
		self.game = Game(self.get_next_shape, self.update_score)
		self.score = Score()
		self.preview = Preview()

		self.input_queue = input_queue  # To receive actions
		self.output_queue = output_queue  # To send state data
		self.state = None

	### UPDATE GAME SCORE
	def update_score(self, lines, score, level):
		self.score.lines = lines
		self.score.score = score
		self.score.level = level

	### GET NEXT SHAPE
	def get_next_shape(self):
		next_shape = self.next_shapes.pop(0)
		self.next_shapes.append(choice(list(TETROMINOS.keys())))
		return next_shape
	
	### INPUT AGENT COMMANDS
	def input_data(self):
		"""Read input actions from the input queue."""
		if self.input_queue and not self.input_queue.empty():
			commands = self.input_queue.get()
			# print(f"Commands: {commands}", flush=True)
            # Simulate the key presses
			self.simulate_key_presses(commands)
			# print(action)
			# if action:
			# 	self.game.process_action(action)\
	
	### SIMULATE KEYPRESS
	def simulate_key_presses(self, commands):
		"""
		Simulate the key presses based on the list of commands.
		:param commands: List of commands to simulate (e.g., ['K_LCTRL', 'K_LEFT']).
		"""
		key_mapping = {
			'K_LCTRL': pygame.K_LCTRL,
			'K_LEFT': pygame.K_LEFT,
			'K_RIGHT': pygame.K_RIGHT,
			'K_DOWN': pygame.K_DOWN,
			'QUIT': pygame.QUIT
		}

		for command in commands:

			if command == 'QUIT':
				# Directly post the QUIT event
				pygame.event.post(pygame.event.Event(pygame.QUIT))
				continue  # Skip further processing for QUIT
			# print(command,flush=True)
			# Map command string to Pygame key
			key = key_mapping.get(command)
			# print(key,flush=True)

			if key:
				# Simulate KEYDOWN
				pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {"key": key}))
				# pygame.time.wait(1)
				# Simulate KEYUP (optional, depends on your game's input handling)
				pygame.event.post(pygame.event.Event(pygame.KEYUP, {"key": key}))

	### OUTPUT STATE TO AGENT
	def output_data(self):
		"""Output the current game state to the output queue."""
		# Create a combined field that includes the current tetromino
		binary = self.game.binary_field.copy()
		combined_field = [row.copy() for row in self.game.binary_field]
		
		if self.game.tetromino and self.game.tetromino.blocks:
			for block in self.game.tetromino.blocks:
				x = int(block.pos.x)
				y = int(block.pos.y)
				if 0 <= y < ROWS and 0 <= x < COLUMNS:
					combined_field[y][x] = 1  # Mark the position of the current tetromino block
			pos = [block.pos for block in self.game.tetromino.blocks]
			shape = self.game.tetromino.blocks[0].shape
			orientation = self.game.tetromino.blocks[0].orientation
		else:
			pos = []
			shape = None
			orientation = None

		# Ensure the reset flag is properly handled
		reset = self.game.reset_flag

		self.state = {
			"combined_field": combined_field,  # Use combined field
			"field_data": binary,
			"score": self.score.score,
			"lines": self.score.lines,
			"pos": pos,
			"shape": shape,
			"orientation": orientation,
			"reset": self.game.reset_flag
		}
		# print(self.game.reset_flag,flush=True)

		if reset:
			self.game.reset_flag = False  # Clear the flag after passing it

		if self.output_queue:
			self.output_queue.put_nowait(self.state)
		return self.state

	def run(self):
		while True:
			for event in pygame.event.get():
				if event.type == pygame.KEYDOWN:
					if event.key == pygame.K_LCTRL:
						# Rotate the piece
						self.game.tetromino.rotate()
					elif event.key == pygame.K_LEFT:
						# Move the piece left
						self.game.tetromino.move_horizontal(-1)
					elif event.key == pygame.K_RIGHT:
						# Move the piece right
						self.game.tetromino.move_horizontal(1)
					elif event.key == pygame.K_DOWN:
						# Move Piece Down
						self.game.tetromino.drop()
				# if event.type == pygame.KEYDOWN:
				# 	# print(f"Key pressed: {event.key}", flush=True)
				# 	# Handle the key press in the game logic
				# if event.type == pygame.KEYUP:
				# 	# print(f"Key released: {event.key}", flush=True)
				if event.type == pygame.QUIT:
					pygame.quit()
					# exit()
					return

			# display 
			self.display_surface.fill(GRAY)
			
			# components
			self.game.run()
			self.score.run()
			self.preview.run(self.next_shapes)

			# Update Queues
			self.output_data()
			self.input_data()

			# updating the game
			pygame.display.update()
			self.clock.tick(120)
			time.sleep(0.001)

if __name__ == '__main__':
	main = Main()
	main.run()