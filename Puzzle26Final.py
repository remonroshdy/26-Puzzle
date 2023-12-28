import copy
import numpy as np
import heapq
import itertools


def read_3d_arrays_from_file(file_path, num_layers, num_rows, num_cols):
    # Initialize the 3D arrays with zeros
    main_3d_array = np.zeros((num_layers, num_rows, num_cols), dtype=int)

    with open(file_path, 'r') as file:
        current_layer = 0
        current_row = 0
        for line in file:
            line = line.strip()
            if not line:
                current_layer += 1
                current_row = 0
            else:
                elements = list(map(int, line.split()))
                main_3d_array[current_layer, current_row] = elements
                current_row += 1

    return main_3d_array


def get_valid_moves(state):
    valid_moves = []

    empty_position = None
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if state[i][j][k] == 0:
                    empty_position = (i, j, k)
                    break

    if empty_position[2] >= 0 and empty_position[2] < 2:
        valid_moves.append("east")

    if empty_position[2] > 0 and empty_position[2] <= 2:
        valid_moves.append("west")

    if empty_position[1] > 0 and empty_position[1] <= 2:
        valid_moves.append("north")

    if empty_position[1] >= 0 and empty_position[1] < 2:
        valid_moves.append("south")

    if empty_position[0] >= 0 and empty_position[0] < 2:
        valid_moves.append("down")

    if empty_position[0] > 0 and empty_position[0] <= 2:
        valid_moves.append("up")

    return valid_moves, empty_position


def successor_states(state, moves, empty_position, steps):
    successor_states = []
    move_mapping = {"east": "E", "west": "W", "north": "N", "south": "S", "up": "U", "down": "D"}

    for move in moves:
        i, j, k = empty_position
        # Create a deep copy of the state
        new_state = copy.deepcopy(state)

        if move == "east":
            new_state[i][j][k], new_state[i][j][k + 1] = new_state[i][j][k + 1], new_state[i][j][k]
            state_empty_position = (i, j, k + 1)

        elif move == "west":
            new_state[i][j][k], new_state[i][j][k - 1] = new_state[i][j][k - 1], new_state[i][j][k]
            state_empty_position = (i, j, k - 1)

        elif move == "north":
            new_state[i][j][k], new_state[i][j - 1][k] = new_state[i][j - 1][k], new_state[i][j][k]
            state_empty_position = (i, j - 1, k)

        elif move == "south":
            new_state[i][j][k], new_state[i][j + 1][k] = new_state[i][j + 1][k], new_state[i][j][k]
            state_empty_position = (i, j + 1, k)

        elif move == "up":
            new_state[i][j][k], new_state[i - 1][j][k] = new_state[i - 1][j][k], new_state[i][j][k]
            state_empty_position = (i - 1, j, k)

        elif move == "down":
            new_state[i][j][k], new_state[i + 1][j][k] = new_state[i + 1][j][k], new_state[i][j][k]
            state_empty_position = (i + 1, j, k)

        successor_states.append((tuple(map(tuple, new_state)), state_empty_position, steps + 1, move_mapping[move]))

    return successor_states


def calculate_heuristic(current_state, goal_state, depth):
    total_manhattan_distance = 0
    print(current_state)
    # Calculate Manhattan distance for each element in the 3D matrix
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Skip the empty tile (value 0)

                value = current_state[i][j][k]
                # exclude empty tile from calculation.
                if value == 0:
                    continue

                goal_position = find_goal_position(value, goal_state)

                distance = manhattan_distance((i, j, k), goal_position)

                total_manhattan_distance += distance

    # Heuristic is the sum of total Manhattan distance
    heuristic = total_manhattan_distance
    return heuristic


# to find for each tile  where it should be in the goal position

def find_goal_position(value, goal_state):
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if goal_state[i][j][k] == value:
                    return (i, j, k)


# to calculate manhattan distance for every tile and ultimately for each state.
def manhattan_distance(position1, position2):
    return sum(abs(p1 - p2) for p1, p2 in zip(position1, position2))


# A* star implementation with priority  queue Min heap and Heap Pop out
def astar(initial_state, goal_state):
    priority_queue = []
    tiebreaker = itertools.count()
    # list to track visited states and their depths
    visited = []
    # to count for the initial state node.
    total_nodes = 1
    # List to store f(n) values for each state
    f_values = []

    # here we are calc heuristic for the initial state and push it into the priority queue
    initial_heuristic = calculate_heuristic(initial_state, goal_state, 0)
    heapq.heappush(priority_queue, (initial_heuristic, next(tiebreaker), initial_state, [], 0))

    # pop out the state with min total distance for expansion
    while priority_queue:
        f_value, _, current_state, path, depth = heapq.heappop(priority_queue)

        depth = len(path)
        f_values.append(f_value)
        # when we reach goal state  we return depth, path , total nodes , and fvalues.
        if np.array_equal(current_state, goal_state):
            return depth, path, total_nodes, f_values

        current_state_str = str(tuple(map(tuple, current_state)))

        if current_state in visited:
            continue
        moves, empty_position = get_valid_moves(current_state)
        successors = successor_states(current_state, moves, empty_position, depth)

        for successor in successors:
            new_state, move, depth, letter_move = successor
            new_path = path + [letter_move]

            # here we are calculating the entire F(n) for each state
            new_f_value = len(new_path) + calculate_heuristic(new_state, goal_state, depth)

            new_state_str = str(new_state)

            # to check if current new state is unique
            if new_state_str not in visited:
                visited.append(new_state_str)
                total_nodes += 1
                # push new unique state to our priority  queue
                heapq.heappush(priority_queue, (new_f_value, next(tiebreaker), new_state, new_path, new_f_value))

    # return none if no solutions
    return None, None, None, None


# write out into our output file
def write_output(output_file, initial_state, goal_state, depth, total_nodes, solution_path, f_values):
    with open(output_file, 'w') as file:
        # Write initial state
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    file.write(str(initial_state[i, j, k]) + ' ')
                file.write('\n')
            file.write('\n')

        # Write goal state
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    file.write(str(goal_state[i, j, k]) + ' ')
                file.write('\n')
            file.write('\n')

        # Write depth and total nodes
        file.write(str(depth) + '\n')
        file.write(str(total_nodes) + '\n')

        # Write solution path
        file.write(' '.join(solution_path) + '\n')

        # Write f values
        file.write(' '.join(map(str, f_values)) + '\n')


def main():
    # reading our file and extract  initial state and goal state.
    file_path = "input3.txt"
    Matrix = read_3d_arrays_from_file(file_path, num_layers=6, num_rows=3, num_cols=3)
    initial_state = Matrix[:3]
    goal_state = Matrix[3:]
    depth, solution_path, total_nodes, _ = astar(initial_state, goal_state)

    if solution_path:
        # calculate f values for the actual path
        f_values = []
        current_state = initial_state
        current_depth = 0

        # add f_value for the initial state
        f_values.append(calculate_heuristic(current_state, goal_state, current_depth))

        for move in solution_path:
            moves, empty_position = get_valid_moves(current_state)
            successors = successor_states(current_state, moves, empty_position, current_depth)

            for successor in successors:
                _, _, depth, letter_move = successor

                if letter_move == move:
                    current_state = successor[0]
                    current_depth = depth
                    f_values.append(current_depth + calculate_heuristic(current_state, goal_state, current_depth))
                    break

        # print out  f_values
        print("f_values for the actual path:", f_values)
        print("Depth:", depth)
        print("Path:", solution_path)
        print("Total nodes:", total_nodes)

        # write our results to output file
        write_output("output.txt", initial_state, goal_state, depth, total_nodes, solution_path, f_values)


if __name__ == "__main__":
    main()
