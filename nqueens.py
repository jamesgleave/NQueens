from multiprocessing import Process, Value, Manager
import random
import time


# Implement a solver that returns a list of queen's locations
#  - Make sure the list is the right length, and uses the numbers from 0 .. BOARD_SIZE-1
def solve(board_size, verbose=0):
    # This almost certainly is a wrong answer!
    board = Board(board_size)
    answer = search(board, verbose)
    return answer


def search(board, verbose):
    # 0 represents searching and 1 represents found
    v = Value('i', 0)
    processes = []
    result = Manager().dict()
    num_processes = min(10, board.board_size)
    for start_col in range(num_processes):
        if verbose == 1:
            print("Start Col:", start_col)
        name = "Process-"+str(start_col)
        # process = Process(target=recursive_solve, args=(board, start_col, v, result, name))
        process = Process(target=iterative_repair_solve, args=(board, v, result, name))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    return [row.index("Q") for row in result["solved"].state]


def recursive_solve(board, col, value, result, name):
    # base case: If all queens are placed
    # then return true

    if value.value == 1:
        return

    if board.is_goal_state(method='backtracking'):
        return True

    # Consider this column and try placing
    # this queen in all rows one by one
    for i in range(board.board_size):
        coord = (i, col % board.board_size)
        if board.check_placement(coord):
            # Place this queen in board[i][col]
            board.place_queen(coord)
            # recur to place rest of the queens
            if recursive_solve(board, col + 1, value, result, name):
                value.value = 1
                result["solved"] = board
                return True

            # If placing queen in board[i][col
            # doesn't lead to a solution, then
            # queen from board[i][col]
            board.backtrace(coord)

    # if the queen can not be placed in any row in
    # this col then return false
    return False


def iterative_repair_solve(board, value=None, result=None, name=None, debug=False, callback=None):

    # first, place all queens
    positions = []
    for col in range(board.board_size):
        row = random.randint(0, board.board_size-1)
        board.place_queen((row, col))
        positions.append(row)

    # iteratively repair
    while not board.is_goal_state(method='iterative_repair'):

        # check if other processes have solved it. If so, return...
        if value is not None and value.value == 1:
            return

        # if a callback is passed, call it!
        if callback is not None:
            callback(board.get_heatmap(show_queens=True, return_type=list))

        # start of bottleneck
        for col in range(board.board_size):
            # Grab the row the queen is in
            row = positions[col]

            # Calculate the conflicts in given column
            # This is a huge bottleneck
            conflicts = [board.get_conflicts((r, col)) for r in range(board.board_size)]

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

    # if a callback is passed, call it!
    if callback is not None:
        callback(board.get_heatmap(show_queens=True, return_type=list))

    # If found, change the value and set the result to solved
    if value is not None:
        value.value = 1
        result["solved"] = board
    print("got the solution fam")
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

    def optimized_get_conflicts(self, queen_coordinates, col_num):
        """
        This method is an optimized version of the get_conflicts method.
        Instead of looking at every point, it only looks where the queens are and calculates the whole conflict matrix.
        Returns a list of columns
        """
        conflict_matrix = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        for pos in queen_coordinates:
            num_rows = self.board_size
            num_cols = self.board_size
            row = num_rows - 1 - pos[0]
            col = pos[1]

            # Finding first diagonal
            diagonal_1_r, diagonal_1_c = num_rows - 1 - max(row - col, 0), max(col - row, 0)
            d1_len = min(diagonal_1_r + 1, num_cols - diagonal_1_c)
            # Find the conflicts in the diagonals
            for k in range(d1_len):
                if not (diagonal_1_r - k, diagonal_1_c + k) == pos:
                    conflict_matrix[diagonal_1_r - k][diagonal_1_c + k] += 1

            # Finding second diagonal
            t = min(row, num_cols - col - 1)
            diagonal_2_r, diagonal_2_c = num_rows - 1 - row + t, col + t
            d2_len = min(diagonal_2_r, diagonal_2_c) + 1
            # Find the conflicts in the diagonals
            for k in range(d2_len):
                if not (diagonal_2_r - k, diagonal_2_c - k) == pos:
                    conflict_matrix[diagonal_2_r - k][diagonal_2_c - k] += 1

            # Now check the row
            for i in range(self.board_size):
                if i != pos[1]:
                    conflict_matrix[pos[0]][i] += 1
        return [row[col_num] for row in conflict_matrix]

    def get_diagonals(self, pos):
        num_rows = self.board_size
        num_cols = self.board_size
        row = num_rows - 1 - pos[0]
        col = pos[1]

        # Finding first diagonal
        diagonal_1_r, diagonal_1_c = num_rows - 1 - max(row - col, 0), max(col - row, 0)
        d1_len = min(diagonal_1_r + 1, num_cols - diagonal_1_c)
        diagonal_1 = [self.state[diagonal_1_r - k][diagonal_1_c + k] for k in range(d1_len)]

        # Finding second diagonal
        t = min(row, num_cols - col - 1)
        diagonal_2_r, diagonal_2_c = num_rows - 1 - row + t, col + t
        d2_len = min(diagonal_2_r, diagonal_2_c) + 1
        diagonal_2 = [self.state[diagonal_2_r - k][diagonal_2_c - k] for k in range(d2_len)]
        return diagonal_1, diagonal_2

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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int)
    parser.add_argument("-multi", type=bool)
    args = parser.parse_args()

    if args.n is not None:
        print("Args passed")
        n = args.n

        print("n =", n)
        print("Using multiprocessing:", args.multi)
        if args.multi:
            solve(n)
        else:
            iterative_repair_solve(Board(n))
    else:
        print("No args passed")
        n = 10
        multi = False

        print("n =", n)
        print("Using multiprocessing:", multi)
        if multi:
            print(solve(n))
        else:
            b = iterative_repair_solve(Board(n))
            print([row.index("Q") for row in b.state])
