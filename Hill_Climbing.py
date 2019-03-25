from Utility import random_start_queens, safe_queens_heuristic_cost, best_neighbour_queens, random_neighbour_queens, best_random_neighbour_queens
import time

class Hill_Climbing():
    '''Implementing hill climbing algorithm'''
    def __init__(self):
        self.repeated_state = 0
        self.num_of_loop = 0
        self.not_found = 1

    def hill_climbing_search(self, n_queen):
        current_queens = random_start_queens(n_queen)
        current_heuristic_cost = 0
        self.num_of_loop = n_queen*10000000
        loop = 0
        while loop < self.num_of_loop:
            # evaluation of the current nodes
            current_heuristic_cost = safe_queens_heuristic_cost(current_queens, n_queen)
            # stochastic neighbours or best neighbours
            neighbour_queens = best_neighbour_queens(current_queens, n_queen, current_heuristic_cost)
            # get the values of evaluation for neighbours
            neighbour_heuristic_cost = safe_queens_heuristic_cost(neighbour_queens, n_queen)
            # compare the values between current nodes and neighbours
            if current_heuristic_cost < neighbour_heuristic_cost:
                # if a new neighbour has a worse evaluation value, return the current queen
                if current_heuristic_cost==0:
                    print("One global optimal solution found.")
                    return current_queens
                else:
                    self.repeated_state += 1
            else:
                current_queens = neighbour_queens.copy()
                self.repeated_state += 1
            # incorporating a random restart method if hill climbing search gets stuck more than two times
            if self.repeated_state >= 3:
                current_queens = random_start_queens(n_queen)
                self.repeated_state = 0
            loop += 1
        print("One local maximum solution found.")
        print("Queens in conflict: %d" % current_heuristic_cost)
        return current_queens