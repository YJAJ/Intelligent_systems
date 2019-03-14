from Breadth_First_Search import Breadth_First_Search
from Queen import Queen
import time

def main():
    #first_queen = Queen(0, 0)
    start_time = time.time()
    solver = Breadth_First_Search()
    result = solver.bfs_search(4)
    print(result)
    duration = time.time()-start_time
    print(duration)
