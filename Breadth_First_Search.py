from collections import deque
from Utility import is_goal_state, total_state, is_queen_safe_col
import time

class Breadth_First_Search():
    ''''Implement breadth first search'''
    def __init__(self):
        self.frontier = deque()
        self.explored = set()
        self.solutions = list()
        self.nSolution = 0
        self.state = 1
        self.queen_position = 0

    def bfs_search(self, n_queen):
        #initial state no queens
        n_queen_square = n_queen**2

        #print out total state
        total_state(n_queen)
        #push row 0 state to frontier queue
        for index in range(n_queen):
            self.frontier.append([(self.queen_position)])
            self.queen_position =  (self.queen_position + 1)%n_queen_square
            self.state += 1

        initial_queens = self.frontier[0]
        if len(initial_queens)==n_queen and is_goal_state(initial_queens, n_queen):
            self.nSolution += 1
            self.solutions.append(initial_queens)

        depth = 2
        last_state = self.state
        log_period = 30

        while depth <= n_queen:
            branch_node = 0
            #branch size = b to the power of the current depth
            #branch_size = n_queen**depth
            branch_size = n_queen
            for i in range(0, depth-1):
                branch_size *= (n_queen-i)
            # if depth == 3:
            #     branch_size = (2 * (n_queen - 2) + (n_queen - 2) * (n_queen - 3)) * n_queen
            # else:
            #     branch_size = n_queen * n_queen

            last_log_time = time.time()

            #
            while branch_node < branch_size and len(self.frontier)!=0:
                # if  len(self.frontier)==0:
                #     return 0

                current_queens = self.frontier.popleft()
                #self.explored.add(hash(tuple(current_node)))
                #print(self.explored)
                queen_position = self.queen_position
                for index in range(n_queen):
                    temp_queens = current_queens.copy()
                    temp_queens.append((queen_position))
                    #if temp_queens not in self.frontier or hash(tuple(temp_queens)) not in self.explored:
                    if len(temp_queens)==n_queen and is_goal_state(temp_queens, n_queen):
                        self.nSolution += 1
                        self.solutions.append(temp_queens)
                    if len(temp_queens) < n_queen and is_queen_safe_col(temp_queens, n_queen):
                        self.frontier.append(temp_queens)
                    queen_position = (queen_position + 1)%n_queen_square
                    self.state += 1
                    # if (time.time() - last_log_time >= log_period):
                    #     states_done = self.state - last_state
                    #     #frontier_bytes = sys.getsizeof(self.frontier) + sys.getsizeof(self.frontier[0])
                    #     print("States checked: %d, %d in last %d seconds, %.2f%% done of total" % (self.state, states_done, log_period, 100. * (float(self.state)/float(total_state))))
                    #     print("Frontier: %d with size of %.6f gb, Explored: %d" % (len(self.frontier), sys.getsizeof(self.frontier)/1000000000, len(self.explored)))
                    #     last_log_time = time.time()
                    #     last_state = self.state
                    branch_node += 1
            self.queen_position = self.queen_position + n_queen
            depth += 1
        print("Number of solutions found: %d" % self.nSolution)
        return self.solutions

