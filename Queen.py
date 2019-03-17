# class Queen():
#     ''''Represents x,y coordinates of the queen(s) in nxn board'''
#     def __init__(self, position):
#         self.q_state = position
#         # self.x_state = x_co
#         # self.y_state = y_co
#
#
#     # def safe_queens_heuristic_cost(self, queens_state, n_queen):
#     #     not_safe = 0
#
#         # rows = [each_node[0] for each_node in queens_state]
#         # columns = [each_node[1] for each_node in queens_state]
#         #
#         # for i in range(len(rows) - 1):
#         #     for j in range(i + 1, len(columns)):
#         #         # check row and column positions of n queens
#         #         if rows[i] == rows[j]:
#         #             not_safe += 1
#         #         if columns[i] == columns[j]:
#         #             not_safe += 1
#         #         # check diagonal positions of n queens
#         #         if abs(rows[i] - rows[j]) == abs(columns[i] - columns[j]):
#         #             not_safe += 1
#
#         for i in range(len(queens_state) - 1):
#             for j in range(i + 1, len(queens_state)):
#                 # check column overlapping
#                 if queens_state[i] % n_queen == queens_state[j] % n_queen:
#                     not_safe += 1
#                 # check diagonal overlapping
#                 if abs(queens_state[i] // n_queen - queens_state[j] // n_queen) == abs(
#                                         queens_state[i] % n_queen - queens_state[j] % n_queen):
#                     not_safe += 1
#                 # check row overlapping
#                 if queens_state[i] // n_queen == queens_state[j] // n_queen:
#                     not_safe += 1
#
#         return not_safe