from collections import deque
from Queen import Queen
from Utility import is_goal_state
import time
import sys

class Breadth_First_Search():
    ''''Implement breadth first search'''
    def __init__(self):
        self.frontier = deque()
        self.explored = set()
        self.nSolution = 0
        self.state = 1
        self.row = 0

    def bfs_search(self, n_queen):
        #initial state no queens
        # if is_goal_state():
        #     self.nSolution += 1

        total_state = 0
        for i in range(n_queen+1):
            total_state += n_queen**i
        print("Total states for %d queens: %d" % (n_queen, total_state))

        #push row 0 state to frontier queue
        for index in range(n_queen):
            queen = Queen(self.row, index)
            self.frontier.append([(queen.x_state, queen.y_state)])
            self.state += 1

        n = 2
        last_state = self.state
        log_period = 30
        while n <= n_queen:
            self.row += 1
            k = 0
            #branch size = b to the power of depth
            branch_size = n_queen**n
            last_log_time = time.time()
            while k < branch_size:
                if  len(self.frontier)==0:
                    return 0

                current_node = self.frontier.popleft()
                #self.explored.add(hash(tuple(current_node)))
                #print(self.explored)
                for index in range(n_queen):
                    temp_node = current_node.copy()
                    child_queen = Queen(self.row, index)
                    temp_node.append((child_queen.x_state, child_queen.y_state))
                    #if temp_node not in self.frontier or hash(tuple(temp_node)) not in self.explored:
                    if len(temp_node)==n_queen and is_goal_state(temp_node):
                        #rint(temp_node)
                        self.nSolution += 1
                    self.frontier.append(temp_node)
                    self.state += 1
                    if (time.time() - last_log_time >= log_period):
                        states_done = self.state - last_state
                        #frontier_bytes = sys.getsizeof(self.frontier) + sys.getsizeof(self.frontier[0])
                        print("States checked: %d, %d in last %d seconds, %.2f%% done of total" % (self.state, states_done, log_period, 100. * (float(self.state)/float(total_state))))
                        print("Frontier: %d with size of %.6f gb, Explored: %d" % (len(self.frontier), sys.getsizeof(self.frontier)/1000000000, len(self.explored)))
                        last_log_time = time.time()
                        last_state = self.state
                    k += 1

            n += 1

        print("States checked: %d, %.2f%% done of total" % (
            self.state, 100. * (float(self.state) / float(total_state))))
        print("Frontier: %d, Explored: %d" % (len(self.frontier), len(self.explored)))

        print(self.state)
        return self.nSolution

