import math
import random
from collections import deque, defaultdict
import heapq
import numpy as np

random.seed(42)

###############################################################################
#                                Node Class                                   #
###############################################################################

class Node:
    """
    Represents a graph node with an undirected adjacency list.
    'value' can store (row, col), or any unique identifier.
    'neighbors' is a list of connected Node objects (undirected).
    """
    def __init__(self, value):
        self.value = value
        self.neighbors = []

    def add_neighbor(self, node):
        """
        Adds an undirected edge between self and node:
         - self includes node in self.neighbors
         - node includes self in node.neighbors (undirected)
        """
        # TODO: Implement adding a neighbor in an undirected manner
        if node not in self.neighbors:
        self.neighbors.append(node)
        if self not in node.neighbors:
        node.neighbors.append(self)
        pass

    def __repr__(self):
        return f"Node({self.value})"
    
    def __lt__(self, other):
        return self.value < other.value


###############################################################################
#                   Maze -> Graph Conversion (Undirected)                     #
###############################################################################

def parse_maze_to_graph(maze):
    """
    Converts a 2D maze (numpy array) into an undirected graph of Node objects.
    maze[r][c] == 0 means open cell; 1 means wall/blocked.

    Returns:
        nodes_dict: dict[(r, c): Node] mapping each open cell to its Node
        start_node : Node corresponding to (0, 0), or None if blocked
        goal_node  : Node corresponding to (rows-1, cols-1), or None if blocked
    """
    rows, cols = maze.shape
    nodes_dict = {}

    # 1) Create a Node for each open cell
    # 2) Link each node with valid neighbors in four directions (undirected)
    # 3) Identify start_node (if (0,0) is open) and goal_node (if (rows-1, cols-1) is open)

    # TODO: Implement the logic to build nodes and link neighbors
    for r in range(rows):
        for c in range(cols):
            if maze[r][c] == 0:  # Open cell
                nodes_dict[(r, c)] = Node((r, c))

    for (r, c), node in nodes_dict.items():
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, Down, Left, Right
            neighbor_pos = (r + dr, c + dc)
            if neighbor_pos in nodes_dict:
                node.add_neighbor(nodes_dict[neighbor_pos])


    start_node = None
    goal_node = None

    # TODO: Assign start_node and goal_node if they exist in nodes_dict
    start_node = nodes_dict.get((0, 0), None)
    goal_node = nodes_dict.get((rows - 1, cols - 1), None)


    return nodes_dict, start_node, goal_node


###############################################################################
#                         BFS (Graph-based)                                    #
###############################################################################

def bfs(start_node, goal_node):
    """
    Breadth-first search on an undirected graph of Node objects.
    Returns a list of (row, col) from start to goal, or None if no path.

    Steps (suggested):
      1. Use a queue (collections.deque) to hold nodes to explore.
      2. Track visited nodes so you don’t revisit.
      3. Also track parent_map to reconstruct the path once goal_node is reached.
    """
    # TODO: Implement BFS
    if not start_node or not goal_node:
        return None

    queue = deque([start_node])
    visited = set()
    parent_map = {start_node: None}

    while queue:
        current = queue.popleft()
        if current == goal_node:
            return reconstruct_path(goal_node, parent_map)
        
        visited.add(current)
        for neighbor in current.neighbors:
            if neighbor not in visited and neighbor not in parent_map:
                parent_map[neighbor] = current
                queue.append(neighbor)
    return None


###############################################################################
#                          DFS (Graph-based)                                   #
###############################################################################

def dfs(start_node, goal_node):
    """
    Depth-first search on an undirected graph of Node objects.
    Returns a list of (row, col) from start to goal, or None if no path.

    Steps (suggested):
      1. Use a stack (Python list) to hold nodes to explore.
      2. Keep track of visited nodes to avoid cycles.
      3. Reconstruct path via parent_map if goal_node is found.
    """
    # TODO: Implement DFS
    if not start_node or not goal_node:
        return None

    stack = [start_node]
    visited = set()
    parent_map = {start_node: None}

    while stack:
        current = stack.pop()
        if current == goal_node:
            return reconstruct_path(goal_node, parent_map)
        
        visited.add(current)
        for neighbor in current.neighbors:
            if neighbor not in visited and neighbor not in parent_map:
                parent_map[neighbor] = current
                stack.append(neighbor)
    return None


###############################################################################
#                    A* (Graph-based with Manhattan)                           #
###############################################################################

def astar(start_node, goal_node):
    """
    A* search on an undirected graph of Node objects.
    Uses manhattan_distance as the heuristic, assuming node.value = (row, col).
    Returns a path (list of (row, col)) or None if not found.

    Steps (suggested):
      1. Maintain a min-heap/priority queue (heapq) where each entry is (f_score, node).
      2. f_score[node] = g_score[node] + heuristic(node, goal_node).
      3. g_score[node] is the cost from start_node to node.
      4. Expand the node with the smallest f_score, update neighbors if a better path is found.
    """
    # TODO: Implement A*
    if not start_node or not goal_node:
        return None

    open_set = []
    heapq.heappush(open_set, (0, start_node))
    g_score = defaultdict(lambda: float('inf'))
    g_score[start_node] = 0
    parent_map = {start_node: None}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal_node:
            return reconstruct_path(goal_node, parent_map)
        
        for neighbor in current.neighbors:
            tentative_g_score = g_score[current] + 1
            if tentative_g_score < g_score[neighbor]:
                parent_map[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + manhattan_distance(neighbor, goal_node)
                heapq.heappush(open_set, (f_score, neighbor))
    return None

def manhattan_distance(node_a, node_b):
    """
    Helper: Manhattan distance between node_a.value and node_b.value 
    if they are (row, col) pairs.
    """
    # TODO: Return |r1 - r2| + |c1 - c2|
    return abs(node_a.value[0] - node_b.value[0]) + abs(node_a.value[1] - node_b.value[1])



###############################################################################
#                 Bidirectional Search (Graph-based)                          #
###############################################################################

def bidirectional_search(start_node, goal_node):
    """
    Bidirectional search on an undirected graph of Node objects.
    Returns list of (row, col) from start to goal, or None if not found.

    Steps (suggested):
      1. Maintain two frontiers (queues), one from start_node, one from goal_node.
      2. Alternate expansions between these two queues.
      3. If the frontiers intersect, reconstruct the path by combining partial paths.
    """
    # TODO: Implement bidirectional search
    if not start_node or not goal_node:
        return None

    # Initialize two frontiers (queues)
    frontier_start = deque([start_node])
    frontier_goal = deque([goal_node])

    # Initialize visited sets and parent maps for both directions
    visited_start = {start_node: None}
    visited_goal = {goal_node: None}

    while frontier_start and frontier_goal:
        # Expand from the start side
        result = expand_bidirectional(frontier_start, visited_start, visited_goal)
        if result:
            return reconstruct_bidirectional_path(result, visited_start, visited_goal)

        # Expand from the goal side
        result = expand_bidirectional(frontier_goal, visited_goal, visited_start)
        if result:
            return reconstruct_bidirectional_path(result, visited_start, visited_goal)
    return None


###############################################################################
#             Simulated Annealing (Graph-based)                               #
###############################################################################

def simulated_annealing(start_node, goal_node, temperature=1.0, cooling_rate=0.99, min_temperature=0.01):
    """
    A basic simulated annealing approach on an undirected graph of Node objects.
    - The 'cost' is the manhattan_distance to the goal.
    - We randomly choose a neighbor and possibly move there.
    Returns a list of (row, col) from start to goal (the path traveled), or None if not reached.

    Steps (suggested):
      1. Start with 'current' = start_node, compute cost = manhattan_distance(current, goal_node).
      2. Pick a random neighbor. Compute next_cost.
      3. If next_cost < current_cost, move. Otherwise, move with probability e^(-cost_diff / temperature).
      4. Decrease temperature each step by cooling_rate until below min_temperature or we reach goal_node.
    """
    # TODO: Implement simulated annealing
    if not start_node or not goal_node:
        return None

    current = start_node
    path = [current.value]
    current_cost = manhattan_distance(current, goal_node)

    while temperature > min_temperature:
        # If the current node is the goal, return the path
        if current == goal_node:
            return path

        # Choose a random neighbor
        if not current.neighbors:
            break
        next_node = random.choice(current.neighbors)

        # Calculate the cost of moving to the next node
        next_cost = manhattan_distance(next_node, goal_node)
        cost_diff = next_cost - current_cost

        # Decide whether to move
        if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
            current = next_node
            path.append(current.value)
            current_cost = next_cost

        # Reduce the temperature
        temperature *= cooling_rate
    return None


###############################################################################
#                           Helper: Reconstruct Path                           #
###############################################################################

def reconstruct_path(end_node, parent_map):
    """
    Reconstructs a path by tracing parent_map up to None.
    Returns a list of node.value from the start to 'end_node'.

    'parent_map' is typically dict[Node, Node], where parent_map[node] = parent.

    Steps (suggested):
      1. Start with end_node, follow parent_map[node] until None.
      2. Collect node.value, reverse the list, return it.
    """
    # TODO: Implement path reconstruction
    path = []
    current = end_node
    while current is not None:
        path.append(current.value)
        current = parent_map[current]
    return path[::-1]


###############################################################################
#                              Demo / Testing                                 #
###############################################################################
if __name__ == "__main__":
    # A small demonstration that the code runs (with placeholders).
    # This won't do much yet, as everything is unimplemented.
    random.seed(42)
    np.random.seed(42)

    # Example small maze: 0 => open, 1 => wall
    maze_data = np.array([
        [0, 0, 1],
        [0, 0, 0],
        [1, 0, 0]
    ])

    # Parse into an undirected graph
    nodes_dict, start_node, goal_node = parse_maze_to_graph(maze_data)
    print("Created graph with", len(nodes_dict), "nodes.")
    print("Start Node:", start_node)
    print("Goal Node :", goal_node)

    # Test BFS (will return None until implemented)
    path_bfs = bfs(start_node, goal_node)
    print("BFS Path:", path_bfs)

    # Similarly test DFS, A*, etc.
    # path_dfs = dfs(start_node, goal_node)
    # path_astar = astar(start_node, goal_node)
    # ...
