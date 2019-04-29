from Breadth_First_Search import Breadth_First_Search
from Breadth_First_Search_Original import Breadth_First_Search_Original
from Hill_Climbing import Hill_Climbing
from Simulated_Annealing import Simulated_Annealing
from Utility import print_solutions
import time

def main():
    start = True
    while start:
        # three algorithms to run: BFS, HC, and SA
        print("Select one of the options from the list below.")
        print("1: Breadth First Search, 2: Hill Climbing, 3: Simulated Annealing, 4: Quit")
        option = input()
        if option=='1':
            option1 = True
            while option1:
                print("Enter the number of queens for your solver:")
                n_queen = int(input())
                if n_queen <= 0:
                    print("Invalid option.")
                else:
                    # measure the time by setting the start time
                    start_time = time.time()
                    # BFS search instance and algorithm
                    solver = Breadth_First_Search()
                    result = solver.bfs_search(n_queen)
                    duration = time.time() - start_time

                    print("Solutions found: %s" % result)
                    print("Time taken: %f seconds" % duration)
                    # giving options for print out on the console
                    if not result==[]:
                        print_solutions(result, n_queen)
                    option1 = False
        elif option=='2':
            option2 = True
            while option2:
                print("Enter the number of queens for your solver:")
                n_queen = int(input())
                if n_queen<=0:
                    print("Invalid option.")
                else:
                    start_time = time.time()
                    solver = Hill_Climbing()
                    result = solver.hill_climbing_search(n_queen)
                    duration = time.time() - start_time
                    # result instead of solution, because it may be a local optimum rather than a solution
                    print("One result found: %s" % result)
                    print("Time taken: %f seconds" % duration)
                    # giving options for print out on the console
                    if result!=None:
                        # print a solution or local maximum
                        print_solutions(result, n_queen)
                    option2 = False
        elif option=='3':
            option3 = True
            while option3:
                print("Enter the number of queens for your solver:")
                n_queen = int(input())
                if n_queen<=0:
                    print("Invalid option.")
                else:
                    start_time = time.time()
                    solver = Simulated_Annealing()
                    result = solver.simulated_annealing_search(n_queen)
                    duration = time.time() - start_time
                    # SA provides a solution
                    print("One solution found: %s" % result)
                    print("Time taken: %f seconds" % duration)
                    # giving options for print out on the console
                    if result!=None:
                        print_solutions(result, n_queen)
                    option3 = False
        elif option=='4':
            start = False
        else:
            print("Invalid option.")