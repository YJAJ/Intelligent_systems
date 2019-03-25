from collections import deque
from Utility import is_goal_state, total_original_state

class Breadth_First_Search_Original():
    ''''Implement original breadth first search without pruning'''
    def __init__(self):
        self.frontier = deque()
        self.explored = set()
        self.solutions = list()
        self.nSolution = 0
        self.state = 1
        self.queen_position = 0

    def bfs_search_original(self, n_queen):
        # initial state has no queens and calculate n queen's squared number (i.e. board size)
        n_queen_square = n_queen**2
        # calculate total state
        total_state = total_original_state(n_queen)
        # push first row's state to frontier queue
        for index in range(n_queen):
            self.frontier.append([self.queen_position])
            self.queen_position =  (self.queen_position + 1)%n_queen_square
            self.state += 1
        # check whether the initial state is the goal state
        initial_queens = self.frontier[0]
        if len(initial_queens)==n_queen and is_goal_state(initial_queens, n_queen):
            self.nSolution += 1
            self.solutions.append(initial_queens)
        # while state is smaller than total state and frontier is not empty, keep the loop
        while self.state < total_state and len(self.frontier)!=0:
            # pop the oldest node and add it to the explored set
            current_queens = self.frontier.popleft()
            self.explored.add(tuple(current_queens))
            # with the popped node, expand the child node
            for position in range(n_queen_square):
                # child's node
                queen_position = position
                # if the child node is not in the current node, the child node can be appended to the list of nodes for later expansion
                if queen_position not in current_queens:
                    temp_queens = current_queens.copy()
                    temp_queens.append(queen_position)
                    temp_queens.sort()
                    # if the expanded nodes are not in frontier or explored,
                    if temp_queens not in self.frontier or tuple(temp_queens) not in self.explored:
                        # check whether the current expanded nodes are the goal state.
                        # if so, add one to solution and append the solution to the solution list.
                        if len(temp_queens)==n_queen and temp_queens not in self.solutions and is_goal_state(temp_queens, n_queen):
                            self.nSolution += 1
                            self.solutions.append(temp_queens)
                        # if the expanded node is not solution, just add this node to frontier and add one to the state
                        self.frontier.append(temp_queens)
                        self.state += 1
                else:
                    # if the child node is already in the list of nodes, add one to the state
                    self.state += 1
            # for memory management, reset explored set to an empty set
            # because there will not be the same list of nodes
            # where the number of nodes inside of the list will be different at the next level.
            self.explored = set()
        # print the total number of solutions found and return the list of solutions
        print("Number of solutions found: %d" % self.nSolution)
        return self.solutions