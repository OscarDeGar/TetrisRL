from settings import *
from sys import exit
from os.path import join

# components
from game import Game
from score import Score
from preview import Preview

from random import choice

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

	def update_score(self, lines, score, level):
		self.score.lines = lines
		self.score.score = score
		self.score.level = level

	def get_next_shape(self):
		next_shape = self.next_shapes.pop(0)
		self.next_shapes.append(choice(list(TETROMINOS.keys())))
		return next_shape
	
	def input_data(self, input):
		input.x

	def output_data(self):
		if self.output_queue:
			state = {
					"field_data": self.game.field_data,
					"score": self.score.score,
					"lines": self.score.lines,
					"shape": self.game.tetromino.shape,
					# "pos": self.game.tetromino.block_positions
			}
			self.output_queue.put(state)

	def run(self):
		while True:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit()
					exit()

			# display 
			self.display_surface.fill(GRAY)
			
			# components
			self.game.run()
			# print(self.game.tetromino.shape)
			# for block in self.game.tetromino.blocks:
			# 	print(block.pos)
			# 	print(block.orientation)
			# print(self.game.tetromino.blocks)
			# print(self.game.sprites	)
			# for row in self.game.field_data:
			# 	for ind in row:
			# 		if ind != 0:
			# 			print(ind.pos)
			self.score.run()
			self.preview.run(self.next_shapes)
			print(self.game.field_data)

			self.output_data()

			# updating the game
			pygame.display.update()
			self.clock.tick()

if __name__ == '__main__':
	main = Main()
	main.run()