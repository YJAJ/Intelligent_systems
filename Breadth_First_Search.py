from collections import deque
from Utility import is_goal_state, total_state, is_queen_safe_col

class Breadth_First_Search():
    '''Implement pruned breadth first search'''
    def __init__(self):
        self.frontier = deque()
        self.explored = set()
        self.solutions = list()
        self.nSolution = 0
        self.state = 1
        self.queen_position = 0

    def bfs_search(self, n_queen):
        # initial state no queens
        n_queen_square = n_queen**2
        # calculate and print out total state
        total_state(n_queen)
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
        # start from level 2 since the empty initial state was level 0 and the one queen on each row was level 1
        depth = 2
        # while depth level is smaller and equal to the number of queen
        while depth <= n_queen:
            branch_node = 0
            # branch size = b to the power of the current depth
            #branch_size = n_queen**depth
            branch_size = n_queen
            for i in range(0, depth-1):
                branch_size *= (n_queen-i)

            # while branch size is smaller than the expected number of states
            while branch_node < branch_size and len(self.frontier)!=0:
                current_queens = self.frontier.popleft()
                # explored set is not required where row separation and column check is undertaken
                #self.explored.add(hash(tuple(current_node)))

                queen_position = self.queen_position
                for index in range(n_queen):
                    temp_queens = current_queens.copy()
                    temp_queens.append(queen_position)
                    # commented out - there will be no equivalent sets given the conditions
                    #if temp_queens not in self.frontier or hash(tuple(temp_queens)) not in self.explored:
                    if len(temp_queens)==n_queen and is_goal_state(temp_queens, n_queen):
                        self.nSolution += 1
                        self.solutions.append(temp_queens)
                    # add the new queens only if queens are safe in a column-wise check
                    if len(temp_queens) < n_queen and is_queen_safe_col(temp_queens, n_queen):
                        self.frontier.append(temp_queens)
                    queen_position = (queen_position + 1)%n_queen_square
                    self.state += 1
                    branch_node += 1
            self.queen_position = self.queen_position + n_queen
            depth += 1
        # print the total number of solutions found and return the list of solutions
        print("Number of solutions found: %d" % self.nSolution)
        return self.solutions