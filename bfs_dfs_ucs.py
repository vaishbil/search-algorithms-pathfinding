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

    # Add comparison methods to prevent errors in priority queue usage
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

# BFS Algorithm with Additional Metrics
def bfs(start, goal, grid):
    start_time = time.time()  # Measure the start time

    queue = deque([start])
    came_from = {start: None}
    visited = {start}
    nodes_explored = 0  # To count nodes explored
    max_queue_size = 1  # For memory usage tracking

    while queue:
        current = queue.popleft()
        nodes_explored += 1  # Increment nodes explored

        if current == goal:
            break

        for neighbor in current.neighbors:
            if neighbor not in visited and not neighbor.is_wall:
                queue.append(neighbor)
                visited.add(neighbor)
                came_from[neighbor] = current

                # Visualize BFS progress
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
        print("No path found!")
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
    
    print(f"BFS completed in {elapsed_time:.4f} seconds.")
    print(f"Memory Usage (Max Queue Size): {max_queue_size} nodes.")
    print(f"Nodes Explored: {nodes_explored}")
    print(f"Path Length: {path_length}")

    return elapsed_time, max_queue_size, nodes_explored, path_length

# DFS Algorithm with Additional Metrics
def dfs(start, goal, grid):
    start_time = time.time()  # Measure the start time

    stack = [start]
    came_from = {start: None}
    visited = {start}  # Track visited nodes immediately when pushing to the stack
    nodes_explored = 0  # To count nodes explored
    max_stack_size = 1  # For memory usage tracking

    while stack:
        current = stack.pop()
        nodes_explored += 1  # Increment nodes explored

        if current == goal:
            break

        for neighbor in current.neighbors:
            if neighbor not in visited and not neighbor.is_wall:
                stack.append(neighbor)
                visited.add(neighbor)  # Mark the neighbor as visited when it's pushed
                came_from[neighbor] = current

                # Visualize DFS progress
                neighbor.color = BLUE
                draw(WIN, grid)
                time.sleep(0.05)  # Slow down the process for real-time visualization

                # Track the maximum size of the stack (memory usage)
                max_stack_size = max(max_stack_size, len(stack))

    # Reconstruct the path
    current = goal
    path = []
    while current:
        path.append(current)
        current = came_from.get(current)

    # If no path is found
    if path[-1] != start:
        print("No path found!")
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
    
    print(f"DFS completed in {elapsed_time:.4f} seconds.")
    print(f"Memory Usage (Max Stack Size): {max_stack_size} nodes.")
    print(f"Nodes Explored: {nodes_explored}")
    print(f"Path Length: {path_length}")

    return elapsed_time, max_stack_size, nodes_explored, path_length

# UCS Algorithm with Additional Metrics
def ucs(start, goal, grid):
    start_time = time.time()  # Measure the start time

    queue = [(0, start)]  # Priority queue with (cost, node)
    came_from = {start: None}
    cost_so_far = {start: 0}
    nodes_explored = 0  # To count nodes explored
    max_queue_size = 1  # For memory usage tracking

    while queue:
        current_cost, current = heapq.heappop(queue)
        nodes_explored += 1  # Increment nodes explored

        if current == goal:
            break

        for neighbor in current.neighbors:
            if not neighbor.is_wall:
                new_cost = current_cost + 1  # All edges have equal cost (1)
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    heapq.heappush(queue, (new_cost, neighbor))
                    came_from[neighbor] = current

                    # Visualize UCS progress
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
        print("No path found!")
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
    
    print(f"UCS completed in {elapsed_time:.4f} seconds.")
    print(f"Memory Usage (Max Queue Size): {max_queue_size} nodes.")
    print(f"Nodes Explored: {nodes_explored}")
    print(f"Path Length: {path_length}")

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
    algorithm = None
    algorithm_name = ""
    run = True
    results = []

    while run:
        draw(WIN, grid)
        font = pygame.font.Font(None, 36)
        if algorithm_name:
            text = font.render(algorithm_name, True, (0, 0, 0))
            WIN.blit(text, (10, 10))
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            # Left Click to Set Start and Goal Separately
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                row, col = pos[1] // CELL_SIZE, pos[0] // CELL_SIZE
                node = grid[row][col]

                if not start:
                    start = node
                    start.color = RED
                    start.is_wall = False
                elif not goal:
                    goal = node
                    goal.color = GREEN
                    goal.is_wall = False

            # Check if the event is a KEYDOWN event
            if event.type == pygame.KEYDOWN:
                # Press '1' to Select BFS Algorithm
                if event.key == pygame.K_1:
                    algorithm = bfs
                    algorithm_name = "BFS"
                # Press '2' to Select DFS Algorithm
                elif event.key == pygame.K_2:
                    algorithm = dfs
                    algorithm_name = "DFS"
                # Press '3' to Select UCS Algorithm
                elif event.key == pygame.K_3:
                    algorithm = ucs
                    algorithm_name = "UCS"
                # Press 'SPACE' to Start the Selected Algorithm
                elif event.key == pygame.K_SPACE and start and goal and algorithm:
                    elapsed_time, memory_usage, nodes_explored, path_length = algorithm(start, goal, grid)
                    results.append((elapsed_time, memory_usage, nodes_explored, path_length))
                    start = None  # Reset start for the next run
                    goal = None  # Reset goal for the next run
                    algorithm_name = ""

            # Press 'S' to Regenerate the Maze
            if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                create_maze(grid)
                start = None  # Reset start for the new maze
                goal = None  # Reset goal for the new maze

    # Print out the results
    print("Results for each run:")
    for i, (time_comp, mem_usage, nodes_exp, path_len) in enumerate(results, 1):
        print(f"Run {i}: Time = {time_comp:.4f} seconds, Memory = {mem_usage} nodes, Nodes Explored = {nodes_exp}, Path Length = {path_len} nodes")

    pygame.quit()

if __name__ == "__main__":
    main()
