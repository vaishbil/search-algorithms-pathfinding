# search-algorithms-pathfinding
python 3.10
Description:
This project demonstrates the working of various AI search algorithms—BFS, DFS, UCS, A*, and Greedy Best-First Search—through an interactive and customizable maze visualizer. Users can modify the maze size, select start and end points, and watch the algorithms in action as they find paths through the maze. The project also compares the performance of these algorithms based on time and space complexity, providing a visual and analytical understanding of their behavior.

Algorithms Included:
1. Breadth-First Search (BFS)
2. Depth-First Search (DFS)
3. Uniform Cost Search (UCS)
4. A* Search Algorithm
5. Greedy Best-First Search

Usage
Run the program:
1. For BFS, DFS, and UCS algorithms, run:

python bfs_dfs_ucs.py

2. For A* and Greedy Best-First Search algorithms, run:

python Astar_greedy.py

Features:
Maze Customization: Users can modify the maze size and manually select the start and end points.
Visualization: Watch the algorithm in action as it traverses the maze.
Performance Metrics: The project can track and display time complexity and space complexity for each algorithm.

Install the required dependencies:
pip install pygame
pip install heapq

Select an Algorithm:
In bfs_dfs_ucs.py:
Press 1 for BFS.
Press 2 for DFS.
Press 3 for UCS.
In Astar_greedy.py:
Press 1 for A*.
Press 2 for Greedy Best-First Search.


Visualize the Algorithm:
After selecting an algorithm, press the "Spacebar" to see the algorithm in action as it traverses the maze.


Color coding:
The start point is Red.
The end point is Green.
The traversal is visualized in Blue as the algorithm explores the maze.
Once the path is found, it backtracks and highlights the final path in Green.


Regenerate the Maze:
Press the "S" key to regenerate the maze and start a new traversal.

![30X30_Maze](https://github.com/user-attachments/assets/d2cb8181-2e01-45c2-aa4e-f5b6fd94cf97)



