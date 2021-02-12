import random
import time


# Implement a solver that returns a list of queen's locations
#  - Make sure the list is the right length, and uses the numbers from 0 .. BOARD_SIZE-1
def solve(board_size, verbose=0, initialization_method='optimal'):

    # This almost certainly is a wrong answer!
    board = Board(board_size)
    answer = search(board, verbose, initialization_method=initialization_method)
    return answer


def search(board, verbose, initialization_method):
    solution = iterative_repair_solve(board, use_iterative_reinitialization=False, debug=False, verbose=verbose, initialization_method=initialization_method)

    # Grab the output
    output = [row.index("Q") for row in solution.state]

    # Transpose it
    transposed = [0 for _ in range(len(output))]
    for i in range(len(transposed)):
        transposed[output[i]] = i

    # Return the list where transposed[col] = row
    return transposed


def iterative_repair_solve(board, value=None, result=None, name="Iterative Repair Bot:", debug=False, callback=None,
                           use_iterative_reinitialization=False, verbose=0, initialization_method='optimal'):

    # Determine if the board can be solved easily...
    if initialization_method == 'optimal':
        # check our condition shown in the paper
        initialization_method = 'random'
        for i in [0, 1, 2, 3]:
            k = (board.board_size - 4 - i) / 6
            if k % 1 == 0.0:
                initialization_method = 'diagonal'
                break

    # first, place all queens
    positions = []
    if initialization_method == 'random':
        if verbose > 0:
            print("Initializing randomly")
        # generate a list of the available rows
        for col in range(board.board_size):
            row = random.randint(0, board.board_size-1)
            board.place_queen((row, col))
            positions.append(row)
    elif initialization_method == 'row_exclusive':
        if verbose > 0:
            print("Initializing row-exclusive")

        # Non random initialization
        available_rows = list(range(board.board_size))
        for col in range(board.board_size):
            row = random.choice(available_rows)
            available_rows.remove(row)
            board.place_queen((row, col))
            positions.append(row)
    elif initialization_method == 'diagonal':
        # Diagonal initialization
        # go from n/2 to n diagonally
        # Place queens from 0 to (n-1)/2)

        # check to see if n/2 is even or odd
        midpoint = round((board.board_size-1)/2) if (round(board.board_size/2)) % 2 == 0 else round(board.board_size/2)
        if verbose > 0:
            print("Initializing diagonally")
            print("Midpoint =", midpoint)

        row = 1
        for col in range(0, midpoint):
            board.place_queen((row, col))
            positions.append(row)
            row += 2

        # Place queens from (n-1)/2) to n
        row = 0
        for col in range(midpoint, board.board_size):
            board.place_queen((row, col))
            positions.append(row)
            row += 2
    else:
        raise ValueError

    # iteratively repair
    if verbose > 0:
        print("Initialized")

    instant_solve = True
    while not board.is_goal_state(method='iterative_repair'):
        total_conflicts = 0
        instant_solve = False
        # LEGACY: check if other processes have a better state, if so, set this state to the best state
        if use_iterative_reinitialization and result is not None:
            items = result.items()

            if "solved" in result.keys():
                return

            items.sort(key=lambda x: x[1][0])
            if len(items) > 0:
                board.state = items[0][1][1]
                positions = items[0][1][2]

        # if a callback is passed, call it!
        if callback is not None:
            callback(board.get_heatmap(show_queens=True, return_type=list))

        # start of bottleneck
        for col in range(board.board_size):
            # check if other processes have solved it. If so, return...
            if value is not None and value.value == 1:
                return

            # Grab the row the queen is in
            row = positions[col]

            # Calculate the conflicts in given column
            # This is a huge bottleneck
            conflicts = [board.get_conflicts((r, col)) for r in range(board.board_size)]

            # Testing this new function. It is slower...
            # conflicts = [board.optimized_get_conflicts((r, col)) for r in range(board.board_size)]

            if debug:
                print("-", " - " * board.board_size)
                print("Initial Queen Position:", (row, col))
                print("Initial Positions:", positions, "\n|")
                print("Initial Board:")
                print("Looking At Column:", col)
                print("v " + " - " * (board.board_size-1) if col == 0 else "- " + " - " * (col-1) + " v " + " - " * (board.board_size - 1 - col))
                print(board)

                print("Calculated conflicts in col", str(col) + ":", conflicts)
                print("Conflict Heatmap:")
                print("v " + " - " * (board.board_size-1) if col == 0 else "- " + " - " * (col-1) + " v " + " - " * (board.board_size - 1 - col))
                print(board.get_heatmap())

            min_val = min(conflicts)
            min_val_index = conflicts.index(min_val)
            # End of bottleneck

            if conflicts[row] != 0:
                # Select new row based on the row.
                # If the conflicts with all duplicates removed is the n long, we have no duplicates
                if len(set(conflicts)) == board.board_size:
                    new_row = min_val_index
                else:
                    # Randomly select the new row from the lowest conflict rows
                    # Get the indices of the lowest values
                    cols = []
                    for i in range(board.board_size):
                        if conflicts[i] == min_val:
                            cols.append(i)
                    new_row = random.choice(cols)

                positions[col] = new_row
                board.swap((row, col), (new_row, col))
                total_conflicts += min_val
            elif debug:
                print("Already in best state")

            if debug:
                print("Updated Positions:", positions)
                print("Updated Board:")
                print("v " + " - " * (board.board_size-1) if col == 0 else "- " + " - " * (col-1) + " v " + " - " * (board.board_size - 1 - col))
                print(board)
                print("Updated Conflicts:", conflicts)
                print("Updated Conflict Heatmap:")
                print("v " + " - " * (board.board_size-1) if col == 0 else "- " + " - " * (col-1) + " v " + " - " * (board.board_size - 1 - col))
                print(board.get_heatmap())
                print("-", " - " * (board.board_size - 1))
                time.sleep(3)
        if verbose > 1:
            print(name, total_conflicts)
        if use_iterative_reinitialization and result is not None:
            result[name] = (total_conflicts, board.state.copy(), positions.copy())

    # if a callback is passed, call it!
    if callback is not None:
        callback(board.get_heatmap(show_queens=True, return_type=list))

    # If found, change the value and set the result to solved
    if value is not None:
        value.value = 1
        result["solved"] = board

    print(name, "got the solution fam." if not instant_solve else "solved it during initialization fam.")
    return board


class Board:
    def __init__(self, board_size):
        self.available = 0
        self.state = [[self.available for _ in range(board_size)] for _ in range(board_size)]
        self.board_size = board_size
        self.num_queens = 0
        self.frames = []

    def is_goal_state(self, method='backtracking'):
        if method == 'iterative_repair':
            # check the position of each queen
            for row_num, row in enumerate(self.state):
                if "Q" in row:
                    coord = row_num, row.index("Q")
                    if self.get_conflicts(coord) > 0:
                        return False
            return True
        else:
            return self.num_queens >= self.board_size

    def check_placement(self, coordinates):
        assert (type(coordinates) is tuple or type(coordinates) is list) and min(coordinates) < len(self.state[0])
        row, col = coordinates
        # We only need to check to the left of each queen
        # Check Diagonal
        d1, d2 = self.get_diagonals(coordinates)
        if "Q" in d1 + d2:
            return False

        # Check row
        if "Q" in self.state[row] and self.state[row][col] != "Q":
            return False

        # Check col
        for r in range(row):
            if self.state[r][col] == "Q" and r != row:
                return False
        return True

    def get_conflicts(self, coordinates):
        # unpack the coordinates and initialize the conflicts
        row, col = coordinates
        conflicts = 0

        # Calculate the diagonals, this is the bottle neck of the bottle neck lol
        d1, d2 = self.get_diagonals(coordinates)

        if self.state[row][col] == "Q":
            # Check diagonal
            queen_count = d1.count("Q")
            if queen_count > 1:
                conflicts += queen_count - 1

            # Check diagonal
            queen_count = d2.count("Q")
            if queen_count > 1:
                conflicts += queen_count - 1

            # Check row
            queen_count = self.state[row].count("Q")
            if queen_count > 1:
                conflicts += queen_count - 1
        else:
            # Check diagonal
            queen_count = d1.count("Q")
            if queen_count > 0:
                conflicts += queen_count

            # Check diagonal
            queen_count = d2.count("Q")
            if queen_count > 0:
                conflicts += queen_count

            # Check row
            queen_count = self.state[row].count("Q")
            if queen_count > 0:
                conflicts += queen_count
        return conflicts

    def get_diagonals(self, pos):
        row = self.board_size - 1 - pos[0]
        col = pos[1]

        # Finding first diagonal
        diagonal_1_r, diagonal_1_c = self.board_size - 1 - max(row - col, 0), max(col - row, 0)
        d1_len = min(diagonal_1_r + 1, self.board_size - diagonal_1_c)

        # Finding second diagonal
        t = min(row, self.board_size - col - 1)
        diagonal_2_r, diagonal_2_c = self.board_size - 1 - row + t, col + t
        d2_len = min(diagonal_2_r, diagonal_2_c) + 1
        return [self.state[diagonal_1_r - k][diagonal_1_c + k] for k in range(d1_len)], [self.state[diagonal_2_r - k][diagonal_2_c - k] for k in range(d2_len)]

    def place_queen(self, coordinates):
        row, col = coordinates
        self.state[row][col] = "Q"
        self.num_queens += 1

    def swap(self, coordinates1, coordinates2):
        r1, c1 = coordinates1[0], coordinates1[1]
        r2, c2 = coordinates2[0], coordinates2[1]
        self.state[r1][c1], self.state[r2][c2] = self.state[r2][c2], self.state[r1][c1]

    def backtrace(self, coordinates):
        row, col = coordinates
        self.state[row][col] = 0
        self.num_queens -= 1

    def get_heatmap(self, show_queens=True, return_type=str):
        conflicts = []
        for r, row in enumerate(self.state):
            if show_queens:
                conf_row = [self.get_conflicts((r, c)) if self.state[r][c] != "Q" else "Q" for c in
                            range(self.board_size)]
            else:
                conf_row = [self.get_conflicts((r, c)) for c in range(self.board_size)]
            conflicts.append(conf_row)

        # Can be returned as a string or as a list
        if return_type is str:
            r = ""
            for row in conflicts:
                for col in row:
                    r += str(col) + "  "
                r += "\n"
            return r
        else:
            return conflicts

    def __repr__(self):
        r = ""
        for row in self.state:
            for col in row:
                r += str(col) + "  "
            r += "\n"
        return r


def generate_time_data(min_n=4, max_n=1000, step_size=1):
    with open("{}_chrono_data.csv".format(time.asctime()), 'a') as data:
        data.write("n,time\n")
        # Run n=4 to n=max_n
        for n in range(min_n, max_n + 1, step_size):
            print("Running n =", n)
            start_time = time.time()
            solve(n, verbose=1, initialization_method='optimal')
            data.write(str(n) + "," + str(time.time()-start_time)+ "\n")
            print()


def manual_execution(n):
    start_time = time.time()
    print("n =", n)
    b = solve(n, verbose=2, initialization_method='optimal')
    end_time = time.time() - start_time
    if end_time < 60:
        print("Time taken:", end_time, "seconds.")
    else:
        print("Time taken:", end_time/60, "minutes.")
    print(b)
