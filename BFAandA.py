#final maze of bfa and A* hybrid algorithm

import pygame
import heapq
from collections import deque
import math
import random
import time  # For adding delay in visualization

# Constants
WIDTH, HEIGHT = 600, 600
ROWS, COLS = 30, 30  # Increased grid size for more complexity
CELL_SIZE = WIDTH // COLS

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (220, 220, 220)
WALL = (50, 50, 50)

# Initialize Pygame
pygame.init()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Maze")


# Node Class to Represent Each Cell
class Node:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.x = col * CELL_SIZE
        self.y = row * CELL_SIZE
        self.color = WHITE
        self.neighbors = []
        self.g_score = float('inf')
        self.f_score = float('inf')
        self.is_wall = False

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, CELL_SIZE, CELL_SIZE))

    def add_neighbors(self, grid):
        if self.row > 0:
            self.neighbors.append(grid[self.row - 1][self.col])  # Up
        if self.row < ROWS - 1:
            self.neighbors.append(grid[self.row + 1][self.col])  # Down
        if self.col > 0:
            self.neighbors.append(grid[self.row][self.col - 1])  # Left
        if self.col < COLS - 1:
            self.neighbors.append(grid[self.row][self.col + 1])  # Right


# Create the Grid
def make_grid():
    grid = []
    for i in range(ROWS):
        grid.append([])
        for j in range(COLS):
            node = Node(i, j)
            grid[i].append(node)
    return grid


# Draw the Grid Lines
def draw_grid(win):
    for i in range(ROWS):
        pygame.draw.line(win, GRAY, (0, i * CELL_SIZE), (WIDTH, i * CELL_SIZE))
    for j in range(COLS):
        pygame.draw.line(win, GRAY, (j * CELL_SIZE, 0), (j * CELL_SIZE, HEIGHT))


# Heuristic Function for A* (Euclidean Distance)
def heuristic(a, b):
    return math.sqrt((a.row - b.row) ** 2 + (a.col - b.col) ** 2)


# A* Search
def a_star_search(start, goal, grid):
    open_set = []
    heapq.heappush(open_set, (0, start.row, start.col, start))
    came_from = {}
    start.g_score = 0
    start.f_score = heuristic(start, goal)

    while open_set:
        current = heapq.heappop(open_set)[3]

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in current.neighbors:
            if neighbor.is_wall:
                continue  # Skip walls
            temp_g_score = current.g_score + 1  # Distance between neighbors is 1
            if temp_g_score < neighbor.g_score:
                came_from[neighbor] = current
                neighbor.g_score = temp_g_score
                neighbor.f_score = neighbor.g_score + heuristic(neighbor, goal)
                if neighbor not in open_set:  # Avoid duplicate entries in open_set
                    heapq.heappush(open_set, (neighbor.f_score, neighbor.row, neighbor.col, neighbor))

        # Visualization step
        current.color = BLUE  # Mark the current node being evaluated
        draw(WIN, grid)
        time.sleep(0.05)  # Adjust delay for visualization speed

    return False


# BFS Search
def bfs_search(start, goal, grid):
    queue = deque([start])
    came_from = {}
    visited = set([start])

    while queue:
        current = queue.popleft()

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in current.neighbors:
            if neighbor not in visited and not neighbor.is_wall:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)

                # Visualization step
                neighbor.color = BLUE  # Mark the neighbor as visited
                draw(WIN, grid)
                time.sleep(0.05)  # Adjust delay for visualization speed

    return False


# Reconstruct the Path
def reconstruct_path(came_from, current):
    while current in came_from:
        current = came_from[current]
        current.color = GREEN  # Change path color to green
        draw(WIN, grid)
        time.sleep(0.05)  # Adjust delay for visualization speed


# Hybrid Search (BFS and A* Switch Based on Heuristic)
def hybrid_search(start, goal, grid, heuristic_threshold=10):
    heuristic_value = heuristic(start, goal)

    if heuristic_value < heuristic_threshold:
        return bfs_search(start, goal, grid)
    else:
        return a_star_search(start, goal, grid)


# Recursive Division Maze Generation with Reduced Dead Ends
def recursive_division(grid, start_row, end_row, start_col, end_col):
    if end_row - start_row < 2 or end_col - start_col < 2:
        return

    if random.choice([True, False]):
        # Horizontal division
        row = random.randint(start_row + 1, end_row - 1)
        for col in range(start_col, end_col + 1):
            if random.random() > 0.3 and grid[row][col] not in [grid[start_row][start_col], grid[end_row][end_col]]:
                grid[row][col].color = WALL
                grid[row][col].is_wall = True
        gap = random.randint(start_col, end_col)
        grid[row][gap].color = WHITE  # Create a gap
        grid[row][gap].is_wall = False
        recursive_division(grid, start_row, row - 1, start_col, end_col)
        recursive_division(grid, row + 1, end_row, start_col, end_col)
    else:
        # Vertical division
        col = random.randint(start_col + 1, end_col - 1)
        for row in range(start_row, end_row + 1):
            if random.random() > 0.3 and grid[row][col] not in [grid[start_row][start_col], grid[end_row][end_col]]:
                grid[row][col].color = WALL
                grid[row][col].is_wall = True
        gap = random.randint(start_row, end_row)
        grid[gap][col].color = WHITE
        grid[gap][col].is_wall = False
        recursive_division(grid, start_row, end_row, start_col, col - 1)
        recursive_division(grid, start_row, end_row, col + 1, end_col)


# Draw Function
def draw(win, grid):
    win.fill(WHITE)

    for row in grid:
        for node in row:
            node.draw(win)

    draw_grid(win)
    pygame.display.update()


# Main Function to Run the Game
def main():
    global grid
    grid = make_grid()

    # Generate a random maze using recursive division
    recursive_division(grid, 0, ROWS - 1, 0, COLS - 1)

    for row in grid:
        for node in row:
            node.add_neighbors(grid)

    start = None
    goal = None

    run = True
    while run:
        draw(WIN, grid)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            # Left Click to Set Start and Goal
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                row, col = pos[1] // CELL_SIZE, pos[0] // CELL_SIZE
                node = grid[row][col]

                if not start:
                    start = node
                    start.color = GREEN
                    start.is_wall = False
                elif not goal:
                    goal = node
                    goal.color = RED
                    goal.is_wall = False

            # Press 'SPACE' to Start the Hybrid Search
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and goal:
                    # Reset node scores before starting
                    for row in grid:
                        for node in row:
                            node.g_score = float('inf')
                            node.f_score = float('inf')

                    start.g_score = 0
                    start.f_score = heuristic(start, goal)

                    # Call hybrid search
                    hybrid_search(start, goal, grid)

                # Press 'S' to shuffle and regenerate the maze
                if event.key == pygame.K_s:
                    start = None
                    goal = None
                    grid = make_grid()
                    recursive_division(grid, 0, ROWS - 1, 0, COLS - 1)
                    for row in grid:
                        for node in row:
                            node.add_neighbors(grid)

    pygame.quit()


if __name__ == "__main__":
    main()
