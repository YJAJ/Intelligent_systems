from collections import deque
from Utility import is_goal_state, total_original_state

class Breadth_First_Search_Original():
    ''''Implement original breadth first search'''
    def __init__(self):
        self.frontier = deque()
        self.explored = set()
        self.solutions = list()
        self.nSolution = 0
        self.state = 1
        self.queen_position = 0

    def bfs_search_original(self, n_queen):
        #initial state has no queens
        n_queen_square = n_queen**2

        #print out total state
        total_state = total_original_state(n_queen)
        #push row 0 state to frontier queue
        for index in range(n_queen):
            self.frontier.append([self.queen_position])
            self.queen_position =  (self.queen_position + 1)%n_queen_square
            self.state += 1

        initial_queens = self.frontier[0]
        if len(initial_queens)==n_queen and is_goal_state(initial_queens, n_queen):
            self.nSolution += 1
            self.solutions.append(initial_queens)

        while self.state < total_state and len(self.frontier)!=0:
            current_queens = self.frontier.popleft()
            self.explored.add(tuple(current_queens))

            for position in range(n_queen_square):
                # child's node
                queen_position = position
                if queen_position not in current_queens:
                    temp_queens = current_queens.copy()
                    temp_queens.append(queen_position)
                    temp_queens.sort()
                    # if
                    if temp_queens not in self.frontier or tuple(temp_queens) not in self.explored:
                        if len(temp_queens)==n_queen and temp_queens not in self.solutions and is_goal_state(temp_queens, n_queen):
                            #print(temp_queens)
                            self.nSolution += 1
                            self.solutions.append(temp_queens)
                        self.frontier.append(temp_queens)
                        self.state += 1
                else:
                    self.state += 1
            self.explored = set()
        print("Number of solutions found: %d" % self.nSolution)
        return self.solutions