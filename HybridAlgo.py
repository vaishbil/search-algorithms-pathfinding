import pygame
import heapq
from collections import deque
import math
import random
import time
import tracemalloc  # For memory usage
import pygame.freetype  # For rendering text

# Constants
WIDTH, HEIGHT = 600, 600
ROWS, COLS = 50, 50  # Grid size
CELL_SIZE = WIDTH // COLS

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (220, 220, 220)
WALL = (50, 50, 50)
YELLOW = (255, 255, 0)  # Highlight for start/goal

# Initialize Pygame
pygame.init()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Maze Pathfinding Visualization")

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

    def __lt__(self, other):
        return self.f_score < other.f_score

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

# Title Screen with Instructions
def title_screen(win):
    font = pygame.freetype.SysFont("Arial", 20)
    win.fill(BLACK)
    title_text = "Maze Pathfinding Visualization"
    instructions = [
        "Left Click: Set Start (Red) and Goal (Green)",
        "Press SPACE: Start Search (BFS or A*)",
        "Press 'S': Reset with Same Walls",
        "Close Window: Exit"
    ]
    font.render_to(win, (WIDTH // 2 - 140, HEIGHT // 4), title_text, WHITE)
    for i, line in enumerate(instructions):
        font.render_to(win, (WIDTH // 2 - 180, HEIGHT // 3 + i * 30), line, GRAY)
    pygame.display.update()
    pygame.time.wait(2000)

# Create the Grid with Nodes
def make_grid():
    grid = []
    for i in range(ROWS):
        grid.append([])
        for j in range(COLS):
            node = Node(i, j)
            grid[i].append(node)
    return grid

# Initialize Walls for the Maze Pattern
def initialize_walls(grid):
    for row in grid:
        for node in row:
            # 30% chance of setting a cell as a wall
            if random.random() < 0.3:
                node.color = WALL
                node.is_wall = True

# Draw Grid Lines
def draw_grid(win):
    for i in range(ROWS):
        pygame.draw.line(win, GRAY, (0, i * CELL_SIZE), (WIDTH, i * CELL_SIZE))
    for j in range(COLS):
        pygame.draw.line(win, GRAY, (j * CELL_SIZE, 0), (j * CELL_SIZE, HEIGHT))

# Heuristic Function for A* (Euclidean Distance)
def heuristic(a, b):
    return math.sqrt((a.row - b.row) ** 2 + (a.col - b.col) ** 2)

# Hybrid Search using BFS for close targets, A* otherwise
def hybrid_search(start, goal, grid, heuristic_threshold=10):
    heuristic_value = heuristic(start, goal)
    if heuristic_value < heuristic_threshold:
        print("Using BFS for this search.")
        path_found, explored_nodes = bfs_search(start, goal, grid)
    else:
        print("Switching from BFS to A* search.")
        path_found, explored_nodes = a_star_search(start, goal, grid)
    
    if not path_found:
        print("No path found.")
    return path_found, explored_nodes

# A* Search
def a_star_search(start, goal, grid):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    start.g_score = 0
    start.f_score = heuristic(start, goal)
    explored_nodes = 0

    while open_set:
        current = heapq.heappop(open_set)[1]
        explored_nodes += 1

        if current == goal:
            path_length = reconstruct_path(came_from, current, grid)
            return True, explored_nodes  # Path found

        for neighbor in current.neighbors:
            if neighbor.is_wall:
                continue
            temp_g_score = current.g_score + 1
            if temp_g_score < neighbor.g_score:
                came_from[neighbor] = current
                neighbor.g_score = temp_g_score
                neighbor.f_score = neighbor.g_score + heuristic(neighbor, goal)
                if neighbor not in open_set:
                    heapq.heappush(open_set, (neighbor.f_score, neighbor))

        current.color = BLUE
        draw(WIN, grid)

    return False, explored_nodes  # No path found

# BFS Search
def bfs_search(start, goal, grid):
    queue = deque([start])
    came_from = {}
    visited = set([start])
    explored_nodes = 0

    while queue:
        current = queue.popleft()
        explored_nodes += 1

        if current == goal:
            path_length = reconstruct_path(came_from, current, grid)
            return True, explored_nodes  # Path found

        for neighbor in current.neighbors:
            if neighbor not in visited and not neighbor.is_wall:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)

                neighbor.color = BLUE
                draw(WIN, grid)

    return False, explored_nodes  # No path found

# Reconstruct the Path
def reconstruct_path(came_from, current, grid):
    path_length = 0
    while current in came_from:
        current = came_from[current]
        current.color = GREEN
        path_length += 1
        draw(WIN, grid)
    return path_length

# Draw Function with Highlights for Start and Goal
def draw(win, grid):
    win.fill(WHITE)
    for row in grid:
        for node in row:
            node.draw(win)
            # Highlight start and goal nodes
            if node.color == RED or node.color == GREEN:
                pygame.draw.rect(win, YELLOW, (node.x, node.y, CELL_SIZE, CELL_SIZE), 2)
    draw_grid(win)
    pygame.display.update()
    time.sleep(0.02)

# Main Function
def main():
    global grid
    grid = make_grid()
    initialize_walls(grid)  # Set the initial wall pattern
    title_screen(WIN)  # Show Title Screen

    # Add neighbors for each node in the grid
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

            # Left Click to Set Start and Goal with Visual Prompt
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                row, col = pos[1] // CELL_SIZE, pos[0] // CELL_SIZE
                node = grid[row][col]

                if not start and not node.is_wall:
                    start = node
                    start.color = RED
                elif not goal and not node.is_wall:
                    goal = node
                    goal.color = GREEN

            # Press 'SPACE' to Start the Hybrid Search
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and goal:
                    # Reset g_score and f_score
                    for row in grid:
                        for node in row:
                            node.g_score = float('inf')
                            node.f_score = float('inf')

                    # Start measurements
                    tracemalloc.start()
                    start_time = time.time()

                    path_found, explored_nodes = hybrid_search(start, goal, grid)

                    # End measurements
                    end_time = time.time()
                    memory_usage = tracemalloc.get_traced_memory()[1] / 1024  # in KB
                    tracemalloc.stop()

                    if path_found:
                        path_length = reconstruct_path({}, goal, grid)
                        print(f"Time taken: {end_time - start_time:.4f} seconds")
                        print(f"Path length: {path_length}")
                        print(f"Nodes explored: {explored_nodes}")
                        print(f"Memory usage: {memory_usage:.2f} KB")
                    else:
                        print("No path found.")

                # Press 'S' to reset and retain the same wall pattern
                if event.key == pygame.K_s:
                    start = None
                    goal = None
                    for row in grid:
                        for node in row:
                            if node.is_wall:
                                node.color = WALL
                            else:
                                node.color = WHITE
                            node.add_neighbors(grid)

    pygame.quit()

if __name__ == "__main__":
    main()
