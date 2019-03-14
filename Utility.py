def is_goal_state(node_state):
    goal_state = True

    rows = [each_node[0] for each_node in node_state]
    columns = [each_node[1] for each_node in node_state]

    # check row and column positions of n queens
    for i in range(len(rows) - 1):
        for j in range(i + 1, len(columns)):
            if rows[i]==rows[j]:
                goal_state = False
                return goal_state
            if columns[i]==columns[j]:
                goal_state = False
                return goal_state
    #check diagonal positions of n queens
    for i in range(len(rows)-1):
        for j in range(i+1,len(columns)):
            if abs(rows[i] - rows[j]) == abs(columns[i] - columns[j]):
                goal_state = False
                return goal_state

    return goal_state
