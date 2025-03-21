from typing import List, Tuple, Dict
import numpy as np
import heapq
from math import sqrt
import matplotlib.pyplot as plt

def create_node(position: Tuple[int, int], g: float = float('inf'), h: float = 0.0, parent: Dict = None) -> Dict:
    return {
        'position': position,
        'g': g,
        'h': h,
        'f': g + h,
        'parent': parent
    }

def calculate_heuristic(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    x1, y1 = pos1
    x2, y2 = pos2
    return abs(x2 - x1) + abs(y2 - y1)  # Manhattan distance for grid

def get_valid_neighbors(grid: np.ndarray, position: Tuple[int, int]) -> List[Tuple[int, int]]:
    x, y = position
    rows, cols = grid.shape
    # Only 4 possible moves (up, down, left, right)
    possible_moves = [
        (x + 1, y), (x - 1, y),  # Down, Up
        (x, y + 1), (x, y - 1)   # Right, Left
    ]
    return [
        (nx, ny) for nx, ny in possible_moves
        if 0 <= nx < rows and 0 <= ny < cols  # Within grid bounds
        and grid[nx, ny] != 1                 # Not an obstacle
    ]

def get_move_cost(grid: np.ndarray, position: Tuple[int, int]) -> int:
    terrain = grid[position[0], position[1]]
    if terrain == 0:  # Free space
        return 1
    elif terrain == 2:  # Mud
        return 3
    elif terrain == 3:  # Rock
        return 5
    else:
        return float('inf')

def reconstruct_path(goal_node: Dict) -> List[Tuple[int, int]]:
    path = []
    current = goal_node
    while current is not None:
        path.append(current['position'])
        current = current['parent']
    return path[::-1]  # Reverse to get path from start to goal

def find_path(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    # Initialize start node
    start_node = create_node(
        position=start,
        g=0,
        h=calculate_heuristic(start, goal)
    )

    # Initialize open list (priority queue) and open dictionary
    open_list = [(start_node['f'], start)]  # Priority queue
    open_dict = {start: start_node}  # For quick lookup of nodes

    # Closed set for explored nodes
    closed_set = set()

    while open_list:
        # Get node with the lowest f value
        _, current_pos = heapq.heappop(open_list)
        current_node = open_dict[current_pos]

        # Check if we've reached the goal
        if current_pos == goal:
            return reconstruct_path(current_node)

        # Add current position to closed set
        closed_set.add(current_pos)

        # Explore neighbors
        for neighbor_pos in get_valid_neighbors(grid, current_pos):
            # Skip if neighbor is already explored
            if neighbor_pos in closed_set:
                continue

            # Calculate tentative g value for this neighbor
            move_cost = get_move_cost(grid, neighbor_pos)
            tentative_g = current_node['g'] + move_cost

            if neighbor_pos not in open_dict:
                # Create a new node for the neighbor
                neighbor = create_node(
                    position=neighbor_pos,
                    g=tentative_g,
                    h=calculate_heuristic(neighbor_pos, goal),
                    parent=current_node
                )
                heapq.heappush(open_list, (neighbor['f'], neighbor_pos))
                open_dict[neighbor_pos] = neighbor
            elif tentative_g < open_dict[neighbor_pos]['g']:
                # Found a better path to this neighbor, update its data
                neighbor = open_dict[neighbor_pos]
                neighbor['g'] = tentative_g
                neighbor['f'] = tentative_g + neighbor['h']
                neighbor['parent'] = current_node
                # Reorder the open list by f value
                heapq.heappush(open_list, (neighbor['f'], neighbor_pos))

    return []  # Return empty list if no path is found

def visualize_path(grid: np.ndarray, path: List[Tuple[int, int]]) -> None:
    # Create a copy of the grid to avoid modifying the original
    grid_copy = np.copy(grid)

    # Mark the path on the grid
    for (x, y) in path:
        grid_copy[x][y] = 8  # Use 8 to represent the path

    # Print the grid
    for row in grid_copy:
        print(''.join(['*' if cell == 8 else str(cell) for cell in row]))

def plot_grid(grid: np.ndarray, path: List[Tuple[int, int]]) -> None:
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the grid
    ax.imshow(grid, cmap='Greys', interpolation='none')

    # Plot the path
    if path:
        path_x = [p[1] for p in path]  # Column (y)
        path_y = [p[0] for p in path]  # Row (x)
        ax.plot(path_x, path_y, color='red', marker='o', markersize=8, linewidth=2, label='Path')

    # Mark start and goal positions
    start = path[0] if path else None
    goal = path[-1] if path else None
    if start:
        ax.plot(start[1], start[0], color='green', marker='s', markersize=10, label='Start')
    if goal:
        ax.plot(goal[1], goal[0], color='blue', marker='s', markersize=10, label='Goal')

    # Add legend
    ax.legend()

    # Show the plot
    plt.show()

def main():
    # Create a grid (20x20), all free space initially
    grid = np.zeros((20, 20))

    # Add obstacles and terrain types
    grid[5:15, 10] = 1  # Vertical wall
    grid[5, 5:15] = 1  # Horizontal wall
    grid[10, 5:10] = 2  # Mud
    grid[15, 15:20] = 3  # Rock

    # Define start and goal positions
    start_pos = (2, 2)
    goal_pos = (18, 18)

    # Find the path
    path = find_path(grid, start_pos, goal_pos)

    if path:
        print(f"Path found with {len(path)} steps and total cost: {sum(get_move_cost(grid, p) for p in path)}")
        visualize_path(grid, path)
        plot_grid(grid, path)
    else:
        print("No path found!")

if __name__ == "__main__":
    main()
