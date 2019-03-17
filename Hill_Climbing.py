import random
from Utility import random_start_queens, safe_queens_heuristic_cost, best_neighbour_queens

class Hill_Climbing():
    ''''Implmenting hill climbing algorithm'''
    def __init__(self):
        self.repeated_state = 0

    def hill_climbing_search(self, n_queen):
        current_queens = random_start_queens(n_queen)

        while True:
            neighbour_queens = best_neighbour_queens(current_queens, n_queen)
            current_heuristic_cost = safe_queens_heuristic_cost(current_queens, n_queen)
            if current_heuristic_cost<=safe_queens_heuristic_cost(neighbour_queens, n_queen):
                if current_heuristic_cost==0:
                    print(current_queens)
                    return current_queens
            else:
                self.repeated_state += 1
                current_queens = neighbour_queens
        # incorporating a random restart method if hill clibming search gets stuck more than three times
            if self.repeated_state >= 10:
                current_queens = random_start_queens(n_queen)
                self.repeated_state = 0