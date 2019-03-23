from Breadth_First_Search import Breadth_First_Search
from Breadth_First_Search_Original import Breadth_First_Search_Original
from Hill_Climbing import Hill_Climbing
from Simulated_Annealing import Simulated_Annealing
from Utility import print_solutions
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

            print("Solutions found: %s" % result)
            print("Time taken: %f seconds" % duration)
            if not result==[]:
                print_solutions(result, n_queen)
        elif option=='2':
            option2 = True
            while option2:
                print("Enter the number of queens for your solver:")
                n_queen = int(input())
                if n_queen<=0:
                    print("Invalid option.")
                    option2 = False
                else:
                    start_time = time.time()
                    solver = Hill_Climbing()
                    result = solver.hill_climbing_search(n_queen)
                    duration = time.time() - start_time
                    print("One result found: %s" % result)
                    print("Time taken: %f seconds" % duration)
                    if result!=None:
                        # print a solution or local maximum
                        print_solutions(result, n_queen)

        elif option=='3':
            option3 = True
            while option3:
                print("Enter the number of queens for your solver:")
                n_queen = int(input())
                if n_queen<=0:
                    print("Invalid option.")
                    option3 = False
                else:
                    start_time = time.time()
                    solver = Simulated_Annealing()
                    result = solver.simulated_annealing_search(n_queen)
                    duration = time.time() - start_time
                    print("One solution found: %s" % result)
                    print("Time taken: %f seconds" % duration)
                    if result!=None:
                        print_solutions(result, n_queen)
                        # print a solution or local maximum
                        print_solutions(result, n_queen)
                        # draw a graph with the number of iteration on x axis and heuristic cost on y axis
        elif option=='4':
            start = False
        else:
            print("Invalid option.")
