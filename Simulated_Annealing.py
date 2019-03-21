import math
import random
import time
from Utility import total_state, random_start_queens, safe_queens_heuristic_cost, random_neighbour_queens, best_neighbour_queens

class Simulated_Annealing():
    ''''Implmenting simulated annealing algorithm'''
    def __init__(self):
        self.temp_max = 30000
        self.temp_min = 3.0
        self.temp_decay_alpha = 0.998
        self.repeat = 0

    def adjust_temp(self, temp, start_time):
        time_passed = time.time()-start_time
        new_temp = temp*(self.temp_decay_alpha**time_passed)
        return new_temp

    def simulated_annealing_search(self, n_queen):
        current_queens = random_start_queens(n_queen)
        temperature = self.temp_max

        # repeated loop is related to the size of neighbourhood, hence, size of n_queen states
        self.repeat = total_state(n_queen)
        repeat_number = 0

        start_time = time.time()

        while repeat_number < self.repeat:
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
            temperature = self.adjust_temp(temperature, start_time)