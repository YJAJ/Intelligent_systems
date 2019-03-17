import math
import random
from Utility import random_start_queens, safe_queens_heuristic_cost, random_neighbour_queens, best_neighbour_queens

class Simulated_Annealing():
    ''''Implmenting simulated annealing algorithm'''
    def __init__(self):
        self.temp_max = 30.0**100
        self.temp_min = 3.0
        self.temp_decay = random.uniform(0, 0.000001)

    def adjust_temp(self, temp):
        new_temp = temp*(1-self.temp_decay)
        return new_temp

    def simulated_annealing_search(self, n_queen):
        current_queens = random_start_queens(n_queen)
        temperature = self.temp_max

        while True:
            if safe_queens_heuristic_cost(current_queens, n_queen) == 0:
                return current_queens
            if temperature<self.temp_min:
                print("Failed to find a solution within the range of the temperature")
                return
            neighbour_queens = best_neighbour_queens(current_queens, n_queen)
            delta_energy = safe_queens_heuristic_cost(neighbour_queens, n_queen) - safe_queens_heuristic_cost(current_queens, n_queen)
            if delta_energy<0:
                current_queens = neighbour_queens
            else:
                # probability is higher when temperature is higher, making a worse move more acceptable at the beginning
                probability = math.exp(-delta_energy/temperature)
                # with the probability, allow current_queens to be worse neighbour_queens
                if random.uniform(0,1)<probability:
                    current_queens = neighbour_queens
            temperature = self.adjust_temp(temperature)