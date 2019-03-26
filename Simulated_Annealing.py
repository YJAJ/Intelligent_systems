import math
import random
import time
from Utility import random_start_queens, safe_queens_heuristic_cost, random_neighbour_queens
from HC_SA_graph import graph_steps_hcost

class Simulated_Annealing():
    '''Implementing simulated annealing algorithm'''
    def __init__(self):
        self.temp_max = 3000000
        self.temp_min = 3.0
        self.temp_decay_alpha = 0.999997
        self.repeat = 0
        self.repeat_number = 0
        self.heuristic_costs = list()
        self.step = 1

    def adjust_temp(self, temp, start_time):
        # get the time elapsed since start time to reflect in the temperature value
        time_elapsed = time.time()-start_time
        # new temperature will be based on the temperature decay alpha to the power of time_elapsed
        new_temp = temp*(self.temp_decay_alpha**time_elapsed)
        return new_temp

    def simulated_annealing_search(self, n_queen):
        # randomly initialise the queens on board
        current_queens = random_start_queens(n_queen)
        # set temperature for the temp max value
        temperature = self.temp_max
        # repeated loop k is related to the size of neighbourhood, hence, size of n_queen states
        self.repeat = n_queen*n_queen*100000000000
        # get the start time so that time elapsed can be traced
        start_time = time.time()
        # while current repeating step is smaller that the total repeating value k
        while self.repeat_number < self.repeat:
            # get the current nodes' heuristic cost
            current_heuristic_cost = safe_queens_heuristic_cost(current_queens, n_queen)
            # evaluate the current nodes to see whether it is the goal state
            if current_heuristic_cost==0:
                self.heuristic_costs.append(safe_queens_heuristic_cost(current_queens, n_queen))
                print("Number of steps taken to find the solution: %d" % self.repeat_number)
                # uncomment below to see the scatter plot for steps taken vs heuristic cost
                #graph_steps_hcost(self.heuristic_costs, self.step)
                return current_queens
            # if adjusted temperature becomes too cool, then stop the annealing process
            if temperature<self.temp_min:
                print("Number of steps taken to fail to find the solution: %d" % self.repeat_number)
                print("Failed to find a solution within the range of the temperature")
                return
            # otherwise, get the heuristic cost for a randomly selected neighbour queen
            neighbour_queens = random_neighbour_queens(current_queens, n_queen)
            # delta energy is the difference between the random neighbour cost minus the current heuristic cost
            delta_energy = safe_queens_heuristic_cost(neighbour_queens, n_queen) - current_heuristic_cost
            # if the heuristic cost of randomly selected neighbour nodes is smaller, then replace current queen with neighbour queens
            if delta_energy<=0:
                current_queens = neighbour_queens
            else:
                # probability is higher when temperature is higher, making a worse move more acceptable at the beginning
                # delta energy is multiplied by 100,000 for a better scale for probability
                probability = math.exp(-delta_energy*100000/temperature)
                # with the previously-determined probability, allow current_queens to be worse neighbour queens
                if random.uniform(0,1)<probability:
                    current_queens = neighbour_queens
            self.step += 1
            # store each step's heuristic cost to visualise in the graph
            self.heuristic_costs.append(safe_queens_heuristic_cost(current_queens, n_queen))
            # adjust temperature, reducing temperature exponentially, gradually
            temperature = self.adjust_temp(temperature, start_time)
            self.repeat_number += 1