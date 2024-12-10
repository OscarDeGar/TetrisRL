from settings import *
from random import choice
from sys import exit
from os.path import join

from timer import Timer

class Game:
	def __init__(self, get_next_shape, update_score):

		# general 
		self.surface = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))
		self.display_surface = pygame.display.get_surface()
		self.rect = self.surface.get_rect(topleft = (PADDING, PADDING))
		self.sprites = pygame.sprite.Group()

		# game connection
		self.get_next_shape = get_next_shape
		self.update_score = update_score

		# lines 
		self.line_surface = self.surface.copy()
		self.line_surface.fill((0,255,0))
		self.line_surface.set_colorkey((0,255,0))
		self.line_surface.set_alpha(120)

	

		# Orientations
		self.orientations = {
            'I': 2,
            'O': 1,
            'T': 4,
            'S': 2,
            'Z': 2,
            'J': 4,
            'L': 4,
        }
		
		# tetromino
		self.field_data = [[0 for x in range(COLUMNS)] for y in range(ROWS)]
		self.binary_field = [[0 for x in range(COLUMNS)] for y in range(ROWS)]
		self.tetromino = Tetromino(
			self.get_next_shape(), 
			self.sprites, 
			self.create_new_tetromino,
			self.field_data,
			self.binary_field,
			self.orientations)

		# timer 
		self.down_speed = UPDATE_START_SPEED
		self.down_speed_faster = self.down_speed * 0.3
		self.down_pressed = False
		self.drop_speed = DROP_SPEED
		self.timers = {
			# 'drop': Timer(self.drop_speed, False, self.drop),
			'vertical move': Timer(self.down_speed, True, self.move_down),
			'horizontal move': Timer(MOVE_WAIT_TIME),
			'rotate': Timer(ROTATE_WAIT_TIME)
		}
		self.timers['vertical move'].activate()

		# score
		self.current_level = 1
		self.current_score = 0
		self.current_lines = 0
		self.reset_flag = False  # Initialize reset flag

	def calculate_score(self, num_lines):
		self.current_lines += num_lines
		self.current_score += SCORE_DATA[num_lines] * self.current_level

		# if self.current_lines / 10 > self.current_level:
		# 	self.current_level += 1
		# 	self.down_speed *= 0.75
		# 	self.down_speed_faster = self.down_speed * 0.3
		# 	self.timers['vertical move'].duration = self.down_speed
			
		self.update_score(self.current_lines, self.current_score, self.current_level)

	def check_game_over(self):
		if self.tetromino is None:
			return  # No tetromino to check, so exit the method
		for block in self.tetromino.blocks:
			if block.pos.y < 4:
				print("Game Over! Resetting...", flush=True)
				self.reset()
				return

	def create_new_tetromino(self, check_game_over=True):
		if check_game_over:
			self.check_game_over()
		self.check_finished_rows()
		self.tetromino = Tetromino(
			self.get_next_shape(),
			self.sprites,
			self.create_new_tetromino,
			self.field_data,
			self.binary_field,
			self.orientations
		)

	def timer_update(self):
		for timer in self.timers.values():
			timer.update()

	def move_down(self):
		self.tetromino.move_down()

	def drop(self):
		self.tetromino.drop()

	def draw_grid(self):

		for col in range(1, COLUMNS):
			x = col * CELL_SIZE
			pygame.draw.line(self.line_surface, LINE_COLOR, (x,0), (x,self.surface.get_height()), 1)

		for row in range(1, ROWS):
			y = row * CELL_SIZE
			pygame.draw.line(self.line_surface, LINE_COLOR, (0,y), (self.surface.get_width(),y))

		self.surface.blit(self.line_surface, (0,0))

	def input(self):
		keys = pygame.key.get_pressed()

		# checking horizontal movement
		if not self.timers['horizontal move'].active:
			if keys[pygame.K_LEFT]:
				self.tetromino.move_horizontal(-1)
				self.timers['horizontal move'].activate()
			if keys[pygame.K_RIGHT]:
				self.tetromino.move_horizontal(1)	
				self.timers['horizontal move'].activate()

		# check for rotation
		if not self.timers['rotate'].active:
			if keys[pygame.K_LCTRL]:
				self.tetromino.rotate()
				self.timers['rotate'].activate()
		
		# Drop action (hard drop)
		if keys[pygame.K_UP]:
			if not self.drop_pressed:
				self.tetromino.drop()
				self.drop_pressed = True
		else:
			self.drop_pressed = False
				
		# down speedup
		if not self.down_pressed and keys[pygame.K_DOWN]:
			self.down_pressed = True
			self.timers['vertical move'].duration = self.down_speed_faster

		if self.down_pressed and not keys[pygame.K_DOWN]:
			self.down_pressed = False
			self.timers['vertical move'].duration = self.down_speed

	def check_finished_rows(self):

		# get the full row indexes 
		delete_rows = []
		for i, row in enumerate(self.field_data):
			if all(row):
				delete_rows.append(i)

		if delete_rows:
			for delete_row in delete_rows:

				# delete full rows
				for block in self.field_data[delete_row]:
					block.kill()

				# move down blocks
				for row in self.field_data:
					for block in row:
						if block and block.pos.y < delete_row:
							block.pos.y += 1

			# rebuild the field data 
			self.field_data = [[0 for x in range(COLUMNS)] for y in range(ROWS)]
			for block in self.sprites:
				self.field_data[int(block.pos.y)][int(block.pos.x)] = block
				self.binary_field[int(block.pos.y)][int(block.pos.x)] = 1

			# update score
			self.calculate_score(len(delete_rows))

	def reset(self):
		# Kill all sprites in the sprites group
		for sprite in self.sprites:
			sprite.kill()

		# Clear the sprites group
		self.sprites.empty()

		# Kill any blocks in field_data and remove references
		for y in range(ROWS):
			for x in range(COLUMNS):
				block = self.field_data[y][x]
				if block:
					block.kill()
					self.field_data[y][x] = 0
					self.binary_field[y][x] = 0

		# Clear field data
		self.field_data = [[0 for _ in range(COLUMNS)] for _ in range(ROWS)]
		self.binary_field = [[0 for _ in range(COLUMNS)] for _ in range(ROWS)]

		# Reset tetromino
		self.tetromino = None

		# Reset game variables
		self.current_level = 1
		self.current_score = 0
		self.current_lines = 0
		self.update_score(0, 0, 1)

		# Set reset flag to True
		self.reset_flag = True
		
		# Reset timers
		self.down_speed = UPDATE_START_SPEED
		self.down_speed_faster = self.down_speed * 0.3
		self.timers['vertical move'].duration = self.down_speed
		self.timers['vertical move'].activate()

		# Clear the surface
		self.surface.fill(GRAY)

		print("Game has been reset!", flush=True)

		# Create a new tetromino
		self.create_new_tetromino()

	def run(self):
		# Update game logic
		self.input()
		self.timer_update()
		self.sprites.update()

		# Clear the surface
		self.surface.fill(GRAY)

		# Draw the current state
		self.sprites.draw(self.surface)
		self.draw_grid()
		self.display_surface.blit(self.surface, (PADDING, PADDING))

		pygame.draw.rect(self.display_surface, LINE_COLOR, self.rect, 2, 2)
		pygame.draw.line(self.display_surface, (255, 0, 0), (20, 180), (420, 180), 2)

class Tetromino:
	def __init__(self, shape, group, create_new_tetromino, field_data, binary_field, orientations):

		# setup 
		self.shape = shape
		self.block_positions = TETROMINOS[shape]['shape']
		self.color = TETROMINOS[shape]['color']
		self.create_new_tetromino = create_new_tetromino
		self.field_data = field_data
		self.binary_field = binary_field

		# create blocks
		self.blocks = [Block(group, pos, self.color, shape, orientations) for pos in self.block_positions]

	# collisions
	def next_move_horizontal_collide(self, blocks, amount):
		collision_list = [block.horizontal_collide(int(block.pos.x + amount), self.field_data) for block in self.blocks]
		return True if any(collision_list) else False

	def next_move_vertical_collide(self, blocks, amount):
		collision_list = [block.vertical_collide(int(block.pos.y + amount), self.field_data) for block in self.blocks]
		return True if any(collision_list) else False

	# movement
	def move_horizontal(self, amount):
		if not self.next_move_horizontal_collide(self.blocks, amount):
			for block in self.blocks:
				block.pos.x += amount

	# drop piece
	def drop(self):
		# Calculate the maximum distance the tetromino can drop
		min_distance = ROWS
		for block in self.blocks:
			x = int(block.pos.x)
			y = int(block.pos.y)
			distance = 0
			for dy in range(1, ROWS - y):
				if y + dy >= ROWS or self.field_data[y + dy][x]:
					break
				distance += 1
			if distance < min_distance:
				min_distance = distance

		# Move all blocks down by the minimum distance
		for block in self.blocks:
			block.pos.y += min_distance

		# Update the field data with the new block positions, ensuring y >= 0
		for block in self.blocks:
			if block.pos.y >= 0:
				self.field_data[int(block.pos.y)][int(block.pos.x)] = block
				self.binary_field[int(block.pos.y)][int(block.pos.x)] = 1

		# Create a new tetromino since the current one has been placed
		self.create_new_tetromino()

	# def move_down(self):
	# 	if not self.next_move_vertical_collide(self.blocks, 1):
	# 		for block in self.blocks:
	# 			block.pos.y += 1
	# 	else:
	# 		for block in self.blocks:
	# 			self.field_data[int(block.pos.y)][int(block.pos.x)] = 1
	# 		self.create_new_tetromino()

	def move_down(self):
		if not self.next_move_vertical_collide(self.blocks, 1):
			for block in self.blocks:
				block.pos.y += 1
		else:
			# Place the blocks in the field_data as integers
			for block in self.blocks:
				if block.pos.y >= 0:
					self.field_data[int(block.pos.y)][int(block.pos.x)] = block
					self.binary_field[int(block.pos.y)][int(block.pos.x)] = 1
			self.create_new_tetromino()

	# rotate
	def rotate(self):
		if self.shape != 'O':

			# 1. pivot point 
			pivot_pos = self.blocks[0].pos

			# 2. new block positions
			new_block_positions = [block.rotate(pivot_pos) for block in self.blocks]

			# 3. collision check
			for pos in new_block_positions:
				# horizontal 
				if pos.x < 0 or pos.x >= COLUMNS:
					return

				# field check -> collision with other pieces
				if self.field_data[int(pos.y)][int(pos.x)]:
					return

				# vertical / floor check
				if pos.y > ROWS:
					return

			# 4. implement new positions
			for i, block in enumerate(self.blocks):
				block.pos = new_block_positions[i]

class Block(pygame.sprite.Sprite):
	def __init__(self, group, pos, color, shape, orientations):
		
		# general
		super().__init__(group)
		self.image = pygame.Surface((CELL_SIZE,CELL_SIZE))
		self.image.fill(color)
		
		# position
		self.orientation = 0
		self.pos = pygame.Vector2(pos) + BLOCK_OFFSET
		self.rect = self.image.get_rect(topleft = self.pos * CELL_SIZE)
		self.shape = shape
		
		self.max_orientations = orientations.get(shape, 1)

	def rotate(self, pivot_pos):

		self.orientation = (self.orientation + 1) % self.max_orientations
		return pivot_pos + (self.pos - pivot_pos).rotate(90)

	def horizontal_collide(self, x, field_data):
		if not 0 <= x < COLUMNS:
			return True

		if field_data[int(self.pos.y)][x]:
			return True

	def vertical_collide(self, y, field_data):
		if y >= ROWS:
			return True

		if y >= 0 and field_data[y][int(self.pos.x)]:
			return True

	def update(self):

		self.rect.topleft = self.pos * CELL_SIZE

	def kill(self):
		super().kill()
		# print(f"Block at position {self.pos} has been killed.", flush=True)