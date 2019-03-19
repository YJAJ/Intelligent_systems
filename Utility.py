import random
import numpy as np

def total_state(no_queen):
    # display the number of state expected - range is from 0 to n_queen level of depth
    total_state = 0
    for i in range(no_queen + 1):
        total_state += no_queen ** i
    print("Total states for %d queens: %d" % (no_queen, total_state))
    return total_state

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

def is_queen_safe_col(queens_state, no_queen):
    safe_state = True

    for i in range(len(queens_state) - 1):
        for j in range(i + 1, len(queens_state)):
            # check column overlapping
            if queens_state[i]%no_queen == queens_state[j]%no_queen:
                safe_state = False
                return safe_state
            # check diagonal overlapping
            # if abs(queens_state[i]//no_queen-queens_state[j]//no_queen)==abs(queens_state[i]%no_queen-queens_state[j]%no_queen):
            #     safe_state = False
            #     return safe_state
    return safe_state

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

def print_solutions(result, n_queen):
    while True:
        print("Would you like to print solution(s)? Y/N")
        print_option = input().lower()
        if print_option.upper()=="Y":
            multi_solutions = False
            if any(isinstance(one_result, list) for one_result in result):
                multi_solutions = True
            make_board(result, n_queen, multi_solutions)
            return
        elif print_option.upper()=="N":
            return
        else:
            print("Invalid option.")

def make_board(result, n_queen, multiple_solutions):
    if not multiple_solutions:
        no_solutions = 1
    else:
        no_solutions = len(result)
        if no_solutions > 10:
            no_solutions = 10
        print("Visualising up to %d result..." % no_solutions)
    for i in range(no_solutions):
        if multiple_solutions:
            sub_result = result[i]
        else:
            sub_result = result
        board = []
        for row in range(n_queen):
            sub_board = []
            for col in range(n_queen):
                if sub_result[row]%n_queen==col:
                    sub_board.append(1)
                else:
                    sub_board.append(0)
            board.append(sub_board)
        for i in board:
            s = '| '
            for j in i:
                if j == 1:
                    s += u'\u2655' + ' '
                else:
                    s += str(j) + ' '
            print (s + '|')
        print(' ' * n_queen * 3)