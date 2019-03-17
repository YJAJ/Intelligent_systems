from Breadth_First_Search import Breadth_First_Search
from Hill_Climbing import Hill_Climbing
from Simulated_Annealing import Simulated_Annealing
import time

def main():
    start = True
    while start:
        print("Select one of the options from the list below.")
        print("1: Breadth First Search, 2: Hill Climbing, 3: Simulated Annealing, 4: Quit")
        option = input()
        if option=='1':
            print("Enter the number of queens for your solver:")
            n_queen = int(input())
            start_time = time.time()
            solver = Breadth_First_Search()
            result = solver.bfs_search(n_queen)
            duration = time.time() - start_time

            print("Total solutions found: %s" % result)
            print("Time taken: %f seconds" % duration)
        elif option=='2':
            print("Enter the number of queens for your solver:")
            n_queen = int(input())
            start_time = time.time()
            solver = Hill_Climbing()
            result = solver.hill_climbing_search(n_queen)
            duration = time.time() - start_time
            print("Total solutions found: %s" % result)
            print("Time taken: %f seconds" % duration)
        elif option=='3':
            print("Enter the number of queens for your solver:")
            n_queen = int(input())
            start_time = time.time()
            solver = Simulated_Annealing()
            result = solver.simulated_annealing_search(n_queen)
            duration = time.time() - start_time

            print("Total solutions found: %s" % result)
            print("Time taken: %f seconds" % duration)
        elif option=='4':
            start = False
        else:
            print("Invalid input.")


