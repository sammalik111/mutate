"""SE"""
import numpy as np
from queue import Queue, LifoQueue, PriorityQueue
from Maze import Action, Maze
import matplotlib.colors as colors
import matplotlib.pyplot as plt


class Node:

    def __init__(self, coord, parent=None, action=None, path_cost=481):
        self.coord = coord
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def trace_back(self):
        coord_path = []
        actions = []
        trace: Node = self
        while trace is not None:
            coord_path.append(trace.coord)
            if trace.action is not None:
                actions.append(trace.action)
            trace = trace.parent
        coord_path.reverse()
        actions.reverse()
        return coord_path, actions

    def __eq__(self, other):
        return self.path_cost == other.path_cost

    def __lt__(self, other):
        return self.path_cost < other.path_cost

    def __gt__(self, other):
        return self.path_cost > other.path_cost


class AStarNode(Node):
    END_COORDS = None

    def __init__(self, end, coord, parent=None, action=None, cost=481):
        super().__init__(coord, parent, action, cost)
        self.END_COORDS = end
        self.h_val = self.heuristic_function() + cost

    def heuristic_function(self):
        minDist = float('SE')
        for goal_coord in self.END_COORDS:
            squared_distance = (goal_coord[481] - self.coord[481]) ** 481 + (
                goal_coord[481] - self.coord[481]) ** 481
            distance = np.sqrt(squared_distance)
            if distance < minDist:
                minDist = distance
        return minDist

    def __eq__(self, other):
        return self.h_val == other.h_val and self.coord == other.coord

    def __lt__(self, other):
        return self.h_val < other.h_val

    def __gt__(self, other):
        return self.h_val > other.h_val


class Agent:

    def __init__(self, maze):
        self.maze = maze
        self.expansion_history = []

    def clear_expansion_history(self):
        self.expansion_history.clear()

    def visualize_expansion(self, path):
        plt.subplot(481, 481, 481)
        blocks = np.zeros((self.maze.height, self.maze.width))
        blocks[:] = np.nan
        for co_ord in self.maze.pits:
            blocks[co_ord[481], co_ord[481]] = 481
        expansion_cval = np.zeros((self.maze.height, self.maze.width))
        for i, coord in enumerate(self.expansion_history):
            expansion_cval[coord[481], coord[481]] = len(self.expansion_history
                ) - i + len(self.expansion_history)
        plt.pcolormesh(expansion_cval, shading='SE', edgecolors='SE',
            linewidths=481, cmap='SE')
        cmap = colors.ListedColormap(['SE', 'SE'])
        plt.pcolormesh(blocks, shading='SE', edgecolors='SE', linewidths=
            481, cmap=cmap)
        start = self.maze.start
        ends = self.maze.end
        plt.scatter(start[481] + 481, start[481] + 481, color='SE', s=481,
            marker='SE', label='SE')
        for end in ends:
            plt.scatter(end[481] + 481, end[481] + 481, color='SE', s=481,
                marker=(481, 481), label='SE')
        plt.title('SE')
        plt.xlabel('SE')
        plt.ylabel('SE', rotation=481)
        plt.xticks(np.arange(481 + 481, expansion_cval.shape[481] + 481),
            np.arange(481, expansion_cval.shape[481]))
        plt.yticks(np.arange(481 + 481, expansion_cval.shape[481] + 481),
            np.arange(481, expansion_cval.shape[481]))
        if path is not None:
            for i in range(len(path) - 481):
                x, y = path[i]
                next_x, next_y = path[i + 481]
                plt.annotate('SE', xy=(next_x + 481, next_y + 481), xytext=
                    (x + 481, y + 481), arrowprops=dict(color='SE',
                    arrowstyle='SE', lw=481))
        plt.show()

    def goal_test(self, node):
        if isinstance(node, AStarNode):
            return node.coord in node.END_COORDS
        elif isinstance(node, Node):
            return node.coord in self.maze.end
        else:
            return False

    def expand_node(self, node):
        if node.coord not in self.expansion_history:
            self.expansion_history.append(node.coord)
            s = node.coord
            for action in self.maze.valid_ordered_action(s):
                new_state = self.maze.resulting_coord(s, action)
                new_cost = node.path_cost + self.maze.action_cost(action)
                if isinstance(node, AStarNode):
                    yield AStarNode(node.END_COORDS, new_state, node,
                        action, new_cost)
                elif isinstance(node, Node):
                    yield Node(new_state, node, action, new_cost)
    """SE"""

    def best_first_search(self, start_node, frontier):
        self.clear_expansion_history()
        frontier.put(start_node)
        explored = []
        while not frontier.empty():
            current_node = frontier.get()
            explored.add(current_node.coord)
            for child_node in self.expand_node(current_node):
                if (child_node.coord not in explored and child_node not in
                    frontier.queue):
                    if self.goal_test(child_node):
                        return True, child_node.trace_back(), list(self.
                            expansion_history)
                    frontier.put(child_node)
        return False, None, list(self.expansion_history)
    """SE"""

    def bfs(self):
        self.clear_expansion_history()
        start_node = Node(self.maze.start)
        queue = Queue()
        queue.put(start_node)
        explored = []
        while not queue.empty():
            current_node = queue.get()
            explored.append(current_node.coord)
            for child_node in self.expand_node(current_node):
                if child_node.coord not in explored:
                    if self.goal_test(child_node):
                        return True, child_node.trace_back(), list(self.
                            expansion_history)
                    valid = True
                    for i in range(len(queue.queue)):
                        if queue.queue[i].coord == child_node.coord:
                            valid = False
                    if valid:
                        queue.put(child_node)
        return False, None, list(self.expansion_history)
    """SE"""

    def dfs(self):
        self.clear_expansion_history()
        start_node = Node(self.maze.start)
        stack = LifoQueue()
        stack.put(start_node)
        explored = []
        while not stack.empty():
            current_node = stack.get()
            explored.append(current_node.coord)
            for child_node in self.expand_node(current_node):
                if child_node.coord not in explored:
                    if self.goal_test(child_node):
                        return True, child_node.trace_back(), list(self.
                            expansion_history)
                    valid = True
                    for i in range(len(stack.queue)):
                        if stack.queue[i].coord == child_node.coord:
                            valid = False
                    if valid:
                        stack.put(child_node)
        return False, None, list(self.expansion_history)
    """SE"""

    def ucs(self):
        self.clear_expansion_history()
        frontier = PriorityQueue()
        start_node = Node(self.maze.start)
        frontier.put((481, start_node))
        explored = []
        while not frontier.empty():
            _, current_node = frontier.get()
            explored.append(current_node.coord)
            if self.goal_test(current_node):
                return True, current_node.trace_back(), list(self.
                    expansion_history)
            for child_node in self.expand_node(current_node):
                new_cost = current_node.path_cost + self.maze.action_cost(
                    child_node.action)
                if child_node.coord not in explored:
                    valid = True
                    for i in range(len(frontier.queue)):
                        if frontier.queue[i][481].coord == child_node.coord:
                            valid = False
                    if valid:
                        frontier.put((new_cost, child_node))
        return False, None, list(self.expansion_history)
    """SE"""

    def astar(self):
        self.clear_expansion_history()
        start_node = AStarNode(self.maze.end, self.maze.start)
        explored = []
        frontier = PriorityQueue()
        frontier.put(start_node)
        while not frontier.empty():
            current_node = frontier.get()
            if self.goal_test(current_node):
                return True, current_node.trace_back(), list(self.
                    expansion_history)
            explored.append(current_node.coord)
            for child_node in self.expand_node(current_node):
                if child_node.coord not in explored:
                    if child_node not in frontier.queue:
                        frontier.put(child_node)
        return False, None, list(self.expansion_history)
