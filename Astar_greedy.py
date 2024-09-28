import pygame
from collections import deque
import heapq
import time  # For performance measurement and slow visualization
import random  # For wall generation

# Constants
WIDTH, HEIGHT = 400, 400
ROWS, COLS = 30, 30  # Smaller grid size for better visualization
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
pygame.display.set_caption("Pathfinding Visualization")

# Node Class to Represent Each Cell
class Node:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.x = col * CELL_SIZE
        self.y = row * CELL_SIZE
        self.color = WHITE
        self.neighbors = []
        self.is_wall = False

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, CELL_SIZE, CELL_SIZE))

    def add_neighbors(self, grid):
        if self.row > 0:  # Up
            self.neighbors.append(grid[self.row - 1][self.col])
        if self.row < ROWS - 1:  # Down
            self.neighbors.append(grid[self.row + 1][self.col])
        if self.col > 0:  # Left
            self.neighbors.append(grid[self.row][self.col - 1])
        if self.col < COLS - 1:  # Right
            self.neighbors.append(grid[self.row][self.col + 1])

    def __lt__(self, other):
        return False  # We don't compare Nodes directly, so always return False


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

# Draw Everything
def draw(win, grid):
    win.fill(WHITE)
    for row in grid:
        for node in row:
            node.draw(win)
    draw_grid(win)
    pygame.display.update()

# Heuristic Function for A* and Greedy Best-First Search (Manhattan Distance)
def heuristic(node1, node2):
    x1, y1 = node1.col, node1.row
    x2, y2 = node2.col, node2.row
    return abs(x1 - x2) + abs(y1 - y2)

# A* Algorithm with Additional Metrics
def a_star(start, goal, grid):
    start_time = time.time()  # Measure the start time

    queue = [(0, start)]  # Priority queue with (f_score, node)
    came_from = {start: None}
    g_score = {start: 0}  # Distance from start
    f_score = {start: heuristic(start, goal)}  # Estimated distance to goal
    nodes_explored = 0  # To count nodes explored
    max_queue_size = 1  # For memory usage tracking

    while queue:
        current = heapq.heappop(queue)[1]
        nodes_explored += 1  # Increment nodes explored

        if current == goal:
            break

        for neighbor in current.neighbors:
            if not neighbor.is_wall:
                temp_g_score = g_score[current] + 1

                if neighbor not in g_score or temp_g_score < g_score[neighbor]:
                    g_score[neighbor] = temp_g_score
                    f_score[neighbor] = temp_g_score + heuristic(neighbor, goal)
                    heapq.heappush(queue, (f_score[neighbor], neighbor))
                    came_from[neighbor] = current

                    # Visualize A* progress
                    neighbor.color = BLUE
                    draw(WIN, grid)
                    time.sleep(0.05)  # Slow down the process for real-time visualization

                    # Track the maximum size of the queue (memory usage)
                    max_queue_size = max(max_queue_size, len(queue))

    # Reconstruct the path
    current = goal
    path = []
    while current:
        path.append(current)
        current = came_from.get(current)

    # If no path is found
    if path[-1] != start:
        font = pygame.font.Font(None, 36)
        text = font.render("No Path Found!", True, RED)
        WIN.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2))
        pygame.display.update()
        time.sleep(2)  # Display message for 2 seconds
        return 0, 0, 0, 0  # Return zero metrics if no path found

    # Mark the final path in green
    for node in path:
        node.color = GREEN
        draw(WIN, grid)

    # Measure the end time and calculate elapsed time
    elapsed_time = time.time() - start_time
    path_length = len(path)

    return elapsed_time, max_queue_size, nodes_explored, path_length

# Greedy Best-First Search (GBFS) Algorithm
def greedy_bfs(start, goal, grid):
    start_time = time.time()  # Measure the start time

    queue = [(0, start)]  # Priority queue with (heuristic, node)
    came_from = {start: None}
    nodes_explored = 0  # To count nodes explored
    max_queue_size = 1  # For memory usage tracking

    while queue:
        current = heapq.heappop(queue)[1]
        nodes_explored += 1  # Increment nodes explored

        if current == goal:
            break

        for neighbor in current.neighbors:
            if not neighbor.is_wall:
                if neighbor not in came_from:
                    heapq.heappush(queue, (heuristic(neighbor, goal), neighbor))
                    came_from[neighbor] = current

                    # Visualize GBFS progress
                    neighbor.color = BLUE
                    draw(WIN, grid)
                    time.sleep(0.05)  # Slow down the process for real-time visualization

                    # Track the maximum size of the queue (memory usage)
                    max_queue_size = max(max_queue_size, len(queue))

    # Reconstruct the path
    current = goal
    path = []
    while current:
        path.append(current)
        current = came_from.get(current)

    # If no path is found
    if path[-1] != start:
        font = pygame.font.Font(None, 36)
        text = font.render("No Path Found!", True, RED)
        WIN.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2))
        pygame.display.update()
        time.sleep(2)  # Display message for 2 seconds
        return 0, 0, 0, 0  # Return zero metrics if no path found

    # Mark the final path in green
    for node in path:
        node.color = GREEN
        draw(WIN, grid)

    # Measure the end time and calculate elapsed time
    elapsed_time = time.time() - start_time
    path_length = len(path)

    return elapsed_time, max_queue_size, nodes_explored, path_length

# Create Maze with Fixed Pattern
def create_maze(grid):
    random.seed(42)  # Set the random seed for consistent maze generation
    for i in range(ROWS):
        for j in range(COLS):
            if random.random() < 0.35:  # Adjust wall density
                grid[i][j].color = BLACK
                grid[i][j].is_wall = True
            else:
                grid[i][j].color = WHITE
                grid[i][j].is_wall = False

    for row in grid:
        for node in row:
            node.add_neighbors(grid)

# Main Function
def main():
    grid = make_grid()
    for row in grid:
        for node in row:
            node.add_neighbors(grid)
    create_maze(grid)

    start = None
    goal = None
    algorithm = None  # Store selected algorithm

    run = True
    while run:
        draw(WIN, grid)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if pygame.mouse.get_pressed()[0]:  # LEFT CLICK
                pos = pygame.mouse.get_pos()
                row, col = pos[1] // CELL_SIZE, pos[0] // CELL_SIZE
                if not start:
                    start = grid[row][col]
                    start.color = RED
                elif not goal:
                    goal = grid[row][col]
                    goal.color = GREEN

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:  # A* Algorithm
                    algorithm = 'a_star'
                    print("A* Algorithm selected.")
                elif event.key == pygame.K_2:  # Greedy Best-First Search
                    algorithm = 'greedy_bfs'
                    print("Greedy Best-First Search selected.")
                elif event.key == pygame.K_SPACE and start and goal:
                    if algorithm == 'a_star':
                        elapsed_time, max_mem, nodes_explored, path_length = a_star(start, goal, grid)
                    elif algorithm == 'greedy_bfs':
                        elapsed_time, max_mem, nodes_explored, path_length = greedy_bfs(start, goal, grid)

                    print(f"Execution Time: {elapsed_time:.4f} seconds")
                    print(f"Max Memory Usage: {max_mem}")
                    print(f"Nodes Explored: {nodes_explored}")
                    print(f"Path Length: {path_length}")

                if event.key == pygame.K_s:  # Regenerate Maze
                    create_maze(grid)
                    start = None
                    goal = None

    pygame.quit()

if __name__ == "__main__":
    main()
