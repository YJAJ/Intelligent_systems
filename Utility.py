from collections import defaultdict
import operator
import random
def is_goal_state(queens_state, no_queen):
    goal_state = True

    for i in range(len(queens_state) - 1):
        for j in range(i + 1, len(queens_state)):
            # check column overlapping
            if queens_state[i]%no_queen == queens_state[j]%no_queen:
                goal_state = False
                return goal_state
            # check diagonal overlapping
            if abs(queens_state[i]//no_queen-queens_state[j]//no_queen)==abs(queens_state[i]%no_queen-queens_state[j]%no_queen):
                goal_state = False
                return goal_state
            # check row overlapping
            if queens_state[i]//no_queen == queens_state[j]//no_queen:
                goal_state = False
                return goal_state

    # rows = [each_node[0] for each_node in node_state]
    # columns = [each_node[1] for each_node in node_state]

    # # check row and column positions of n queens
    # for i in range(len(rows) - 1):
    #     for j in range(i + 1, len(columns)):
    #         if rows[i]==rows[j]:
    #             goal_state = False
    #             return goal_state
    #         if columns[i]==columns[j]:
    #             goal_state = False
    #             return goal_state
    # #check diagonal positions of n queens
    # for i in range(len(rows)-1):
    #     for j in range(i+1,len(columns)):
    #         if abs(rows[i] - rows[j]) == abs(columns[i] - columns[j]):
    #             goal_state = False
    #             return goal_state

    return goal_state

def random_start_queens(n_queen):
    queens = []
    for row_index in range(n_queen):
        lower = n_queen*(row_index)
        upper = n_queen*(row_index+1)-1
        random_index = random.randint(lower, upper)
        queens.append(random_index)
    return queens

def safe_queens_heuristic_cost(queens_state, no_queen):
    not_safe = 0

    for i in range(len(queens_state) - 1):
        for j in range(i + 1, len(queens_state)):
            # check column overlapping
            if queens_state[i] % no_queen == queens_state[j] % no_queen:
                not_safe += 1
            # check diagonal overlapping
            if abs(queens_state[i] // no_queen - queens_state[j] // no_queen) == abs(
                                    queens_state[i] % no_queen - queens_state[j] % no_queen):
                not_safe += 1
            # check row overlapping
            if queens_state[i] // no_queen == queens_state[j] // no_queen:
                not_safe += 1

    return not_safe

def best_neighbour_queens(current_queens, no_queen):
    current_neighbour = current_queens
    current_heuristic_cost = safe_queens_heuristic_cost(current_queens, no_queen)
    neighbour_generator = 0
    while neighbour_generator < 100:
        #randomly move columns for the queen on each row2
        for row_index in range(no_queen):
            left = current_queens[row_index]
            right = current_queens[row_index]
            if current_queens[row_index]%no_queen!=0:
                left = current_queens[row_index]-1
            if current_queens[row_index]%no_queen!=no_queen-1:
                right = current_queens[row_index]+1
            current_neighbour[row_index] = random.choice([left, right])
        new_heuristic_cost = safe_queens_heuristic_cost(current_neighbour, no_queen)
        neighbour_generator += 1
        if new_heuristic_cost < current_heuristic_cost:
            current_heuristic_cost = new_heuristic_cost
            new_queens = current_neighbour
            return new_queens

    return current_queens

def random_neighbour_queens(current_queens, no_queen):
    current_neighbour = current_queens
    #randomly move columns for the queen on each row
    for row_index in range(no_queen):
        left = current_queens[row_index]
        right = current_queens[row_index]
        if current_queens[row_index]%no_queen!=0:
            left = current_queens[row_index]-1
        if current_queens[row_index]%no_queen!=no_queen-1:
            right = current_queens[row_index]+1
        current_neighbour[row_index] = random.choice([left, right])
    new_queens = current_neighbour
    return new_queens

def print_solutions():
    print()
