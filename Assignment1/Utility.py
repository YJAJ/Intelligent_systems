import random
import operator
from collections import defaultdict

def total_original_state(no_queen):
    # display the number of state expected
    total_original_state = (no_queen*no_queen)**no_queen
    return total_original_state

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
            # check row overlapping - unlikely to happen for pruned models
            if queens_state[i]//no_queen == queens_state[j]//no_queen:
                goal_state = False
                return goal_state

    return goal_state

def is_queen_safe_col(queens_state, no_queen):
    # same with the function above but only for column check
    safe_state = True

    for i in range(len(queens_state) - 1):
        for j in range(i + 1, len(queens_state)):
            # check column overlapping
            if queens_state[i]%no_queen == queens_state[j]%no_queen:
                safe_state = False
                return safe_state
    return safe_state

def random_start_queens(n_queen):
    # initialise queens randomly on each row
    queens = []
    for row_index in range(n_queen):
        lower = n_queen*(row_index)
        upper = n_queen*(row_index+1)-1
        random_index = random.randint(lower, upper)
        queens.append(random_index)
    return queens

def safe_queens_heuristic_cost(queens_state, no_queen):
    # similar to check whether the nodes are in the goal state, but in this case calculating queens in conflict
    not_safe = 0

    for i in range(len(queens_state) - 1):
        for j in range(i + 1, len(queens_state)):
            # check column conflicts
            if queens_state[i] % no_queen == queens_state[j] % no_queen:
                not_safe += 1
            # check diagonal conflicts
            if abs(queens_state[i] // no_queen - queens_state[j] // no_queen) == abs(
                                    queens_state[i] % no_queen - queens_state[j] % no_queen):
                not_safe += 1
            # check row conflicts
            if queens_state[i] // no_queen == queens_state[j] // no_queen:
                not_safe += 1

    return not_safe

def best_random_neighbour_queens(current_queens, no_queen, current_heuristic_cost):
    # if there is only one node
    if len(current_queens)==1:
        return current_queens
    current_neighbours = list()
    # make the list of random neighbours
    for row_index in range(no_queen*5):
        current_neighbour = random_start_queens(no_queen)
        current_neighbours.append(current_neighbour.copy())

    # select the neighbour with the lowest heuristic cost
    best_random_neighbour = select_best_neighbour(current_neighbours, no_queen)
    return best_random_neighbour

def best_neighbour_queens(current_queens, no_queen, current_heuristic_cost):
    if len(current_queens)==1:
        return current_queens
    current_neighbours = list()
    # make a combination of one column move for the queen on each row
    for row_index in range(no_queen):
        current_neighbour = current_queens.copy()
        col_set = set()
        for i in range(no_queen):
            col_set.add(i)
        current_col = current_queens[row_index]%no_queen
        col_set.remove(current_col)
        for remainder in col_set:
            new_position = row_index*no_queen+remainder
            current_neighbour[row_index] = new_position
            current_neighbours.append(current_neighbour.copy())

    # select the neighbour with the lowest heuristic cost
    best_neighbour = select_best_neighbour(current_neighbours, no_queen)
    return best_neighbour

def select_best_neighbour(current_neighbours, no_queen):
    neighbours_dict = defaultdict()
    # calculate and sort different neighbours by their heuristic cost
    for current_neighbour in current_neighbours:
        new_heuristic_cost = safe_queens_heuristic_cost(current_neighbour, no_queen)
        neighbours_dict[new_heuristic_cost] = current_neighbour
    # select the list of queens with the minimum heuristic cost
    best_neighbour= min(neighbours_dict.items(), key = operator.itemgetter(0))[1]
    return best_neighbour

def random_neighbour_queens(current_queens, no_queen):
    if len(current_queens)==1:
        return current_queens
    current_neighbours = list()
    # make a combination of one column move for the queen on each row
    for row_index in range(no_queen):
        current_neighbour = current_queens.copy()
        col_set = set()
        for i in range(no_queen):
            col_set.add(i)
        current_col = current_queens[row_index]%no_queen
        col_set.remove(current_col)
        for remainder in col_set:
            new_position = row_index*no_queen+remainder
            current_neighbour[row_index] = new_position
            current_neighbours.append(current_neighbour.copy())
    # select a random neighbour from the list
    random_neighbour = random.choice(current_neighbours)
    return random_neighbour

def print_solutions(result, n_queen):
    while True:
        print("Would you like to print solution(s)? Y/N")
        print_option = input().lower()
        # print solution if yes
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
    # making a board to visualise queens in specific positions
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