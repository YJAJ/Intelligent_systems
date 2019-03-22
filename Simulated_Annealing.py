import math
import random
import time
from Utility import total_state, random_start_queens, safe_queens_heuristic_cost, random_neighbour_queens, best_neighbour_queens

class Simulated_Annealing():
    ''''Implmenting simulated annealing algorithm'''
    def __init__(self):
        self.temp_max = 3000000000
        self.temp_min = 3.0
        self.temp_decay_alpha = -math.log(self.temp_max/self.temp_min)
        self.repeat = 0
        self.repeat_number = 0

    def adjust_temp(self, temp, start_time):
        time_elapsed = time.time()-start_time
        #print(self.temp_decay_alpha)
        new_temp = temp*math.exp(self.temp_decay_alpha*self.repeat_number/self.repeat)
        return new_temp

    def simulated_annealing_search(self, n_queen):
        current_queens = random_start_queens(n_queen)
        temperature = self.temp_max

        # repeated loop is related to the size of neighbourhood, hence, size of n_queen states
        self.repeat = n_queen*n_queen*1000
        start_time = time.time()

        while self.repeat_number < self.repeat:
            current_heuristic_cost = safe_queens_heuristic_cost(current_queens, n_queen)
            if current_heuristic_cost==0:
                # draw a graph with the number of iteration on x axis and heuristic cost on y axis
                return current_queens
            if temperature<self.temp_min:
                print("Failed to find a solution within the range of the temperature")
                # draw a graph with the number of iteration on x axis and heuristic cost on y axis
                return
            neighbour_queens = random_neighbour_queens(current_queens, n_queen)
            delta_energy = safe_queens_heuristic_cost(neighbour_queens, n_queen) - current_heuristic_cost
            if delta_energy<0:
                current_queens = neighbour_queens
            else:
                # probability is higher when temperature is higher, making a worse move more acceptable at the beginning
                # print(temperature)
                probability = math.exp(-delta_energy/temperature)
                # print(probability)
                # with the probability, allow current_queens to be worse neighbour_queens
                if random.uniform(0,1)<probability:
                    current_queens = neighbour_queens
            temperature = self.adjust_temp(temperature, start_time)
            self.repeat_number += 1