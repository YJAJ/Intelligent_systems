from Utility import random_start_queens, safe_queens_heuristic_cost, best_neighbour_queens
from HC_SA_graph import graph_steps_hcost


class Hill_Climbing():
    '''Implementing modified hill climbing algorithm'''
    def __init__(self):
        self.repeated_state = 0
        self.num_of_loop = 0
        self.heuristic_costs = list()
        self.step = 1

    def hill_climbing_search(self, n_queen):
        current_queens = random_start_queens(n_queen)
        current_heuristic_cost = 0
        # determine the number of loop to be undertaken by HC search
        self.num_of_loop = n_queen*10000000
        loop = 0
        while loop < self.num_of_loop:
            # evaluation of the current nodes based on the heuristic cost (i.e. conflicts in queens)
            current_heuristic_cost = safe_queens_heuristic_cost(current_queens, n_queen)
            # best neighbours (default) or stochastic neighbours
            neighbour_queens = best_neighbour_queens(current_queens, n_queen, current_heuristic_cost)
            # get the values of evaluation for the best neighbour
            neighbour_heuristic_cost = safe_queens_heuristic_cost(neighbour_queens, n_queen)
            # compare the values between current queens and neighbour queens
            # if a new neighbour has a worse evaluation value,
            if current_heuristic_cost < neighbour_heuristic_cost:
                # if the heuristic cost for current queens is equivalent to zero, it is global optimal solution
                if current_heuristic_cost==0:
                    self.heuristic_costs.append(safe_queens_heuristic_cost(current_queens, n_queen))
                    print("One global optimal solution found.")
                    print("Number of steps taken to find the solution: %d" % self.step)
                    # uncomment below to see the scatter plot for steps taken vs heuristic cost
                    #graph_steps_hcost(self.heuristic_costs, self.step)
                    return current_queens
                else:
                # otherwise, count the repeated (i.e. stuck) condition
                    self.repeated_state += 1
            else:
                # if a new neighbour has a better heuristic cost, replace current queens with neighbour queens
                current_queens = neighbour_queens.copy()
                self.repeated_state += 1
            # incorporating a random restart method if hill climbing search gets stuck more than two times
            if self.repeated_state >= 3:
                current_queens = random_start_queens(n_queen)
                self.repeated_state = 0
            self.step += 1
            loop += 1
            # store each step's heuristic cost to visualise in the graph
            self.heuristic_costs.append(safe_queens_heuristic_cost(current_queens, n_queen))
        # if the global optimal solution was not found by the end of the loop, just return the local maximum solution
        print("One local maximum solution found.")
        print("Number of steps taken to fail to find the solution: %d" % self.step)
        print("Queens in conflict: %d" % current_heuristic_cost)
        return current_queens