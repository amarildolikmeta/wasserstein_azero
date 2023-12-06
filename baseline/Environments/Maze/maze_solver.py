import numpy as np


def get_cell_value(maze, coords: tuple):
    """
    Gets the value of the cell at the specified coordinates

    :param coords: tuple containing x and y values
    :return: value of the cell at the specifed coordinates
    """
    try:
        if coords[0] < 0 or coords[0] >= len(maze) or coords[1] < 0 or coords[1] >= len(maze):
            raise IndexError
        return maze[coords[0]][coords[1]]
    # Sometimes we get an IndexError if the maze doesn't have borders
    # This solution is not perfect however, so it is still best practice to use borders
    except IndexError:
        return False


def get_cell_neighbours(maze, coords: tuple, mode="normal"):
    """
    Gets the values of all cells that neighbour the cell at the specified coordinates

    :param coords: Tuple containing the x and y values of the cell to check the neighbours of
    :param mode: specifies whether we are doing our first pass or backtracking from
                 the exit. Is either "normal" (default) or "backtrack"
    :return: coordinates of all neighbours that have not been visited in
                a list of tuples. Example: [(x,y), (x,y), (x,y)]
    """
    # different tuples that contain the coords of all positions
    # relative to our input tuple
    left = (coords[0], coords[1] - 1)
    right = (coords[0], coords[1] + 1)
    up = (coords[0] - 1, coords[1])
    down = (coords[0] + 1, coords[1])

    # list containing all directional tuples
    all_dirs = [left, right, up, down]
    visitable_coordinates = []

    if mode == "normal":
        for dir in all_dirs:
            cell_value = get_cell_value(maze, dir)

            if cell_value == ".":  # if unvisited path
                visitable_coordinates.append(dir)

    elif mode == "backtrack":  # if we are backtracking
        for dir in all_dirs:
            cell_value = get_cell_value(maze, dir)

            if type(cell_value) == int:  # if path has been visited
                visitable_coordinates.append(dir)

    return visitable_coordinates


def get_cells_by_value(maze, value):
    """
    Get cell coordinates based on the value of the cell.

    :param value: The value to search cells for
    :return: list of all coordinates that contain the specified value
    """
    all_matching_cells = []  # the list containing all the coordinates of cells
    for row_index, row in enumerate(maze):
        for column_index, cell in enumerate(row):
            if cell == value:
                all_matching_cells.append((row_index, column_index))

    return all_matching_cells


def get_cell_by_value(value):
    """
    The same as get_cells_by_value, except raises a ValueError if there is more than one cell with that value

    :param value: The value to search cells for
    :raises ValueError: If more then one of the value is found in the maze.
    :return: the cell coordinate that contains the value
    """
    values = get_cells_by_value(value)
    if len(values) > 1:
        raise ValueError(f"Expected only one cell to have value '{value}'. {len(values)} cells contained the value.")

    return values[0]


def set_cell_value(distances, coords: tuple, value: str or int):
    """
    Sets the value of a cell at the specified coordinates.

    :param coords: The coordinates of the cell to be changed
    :param value: The value we want the cell to be set to
    """
    distances[coords[0]][coords[1]] = value


def get_final_path(maze, end_pos: tuple):
    """
    Starts at the exit of the maze and works backwards to the entrance.

    :return: a list of all the cell coordinates that make up the path from the exit to the entrance
    """
    reverse_final_path = []  # stores a path from the exit to the entrance

    current_cell = end_pos  # stores the cell we are currently at

    reverse_final_path.append(current_cell)

    dist_from_start = get_cell_value(maze, end_pos)  # the distance from the entrance

    latest_row = len(maze) - 1
    while dist_from_start >= 0:
        neighbours = get_cell_neighbours(maze, current_cell, mode="backtrack")
        for coords in neighbours:
            if coords[0] == latest_row - 1:
                latest_row -= 1

            if maze[coords[0]][coords[1]] == dist_from_start - 1:
                current_cell = (coords[0], coords[1])
                reverse_final_path.append(coords)
                break

        dist_from_start = dist_from_start - 1

    return reverse_final_path


def pi(state, three_d=False):
    """
    Main entrypoint that solves the maze and outputs a list of cells representing a solution

    :return: A list of tuples (x, y) representing the cells in the solution path
    """
    empty_cell = 0
    occupied_cell = 1
    current_cell = 2
    goal_cell = 3
    state = np.squeeze(state)
    start_dist = 0  # the distance from the entrance
    maze = np.array(state)
    maze = maze * 3
    start_pos = np.where(maze == current_cell)
    start_pos = (start_pos[0][0], start_pos[1][0])
    end_pos = np.where(maze == goal_cell)
    end_pos = (end_pos[0][0], end_pos[1][0])
    maze = maze.tolist()
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i][j] == empty_cell:
                maze[i][j] = '.'
            elif maze[i][j] == occupied_cell:
                maze[i][j] = '#'
            elif maze[i][j] == current_cell:
                maze[i][j] = 's'
            elif maze[i][j] == goal_cell:
                maze[i][j] = 'e'

    set_cell_value(maze, start_pos, 0)  # mark the entrance as visited
    # progress_bar = progress.bar.PixelBar(g.change_string_length("Enumerating maze", 30), max=len(g.maze) - 1)
    current_progress = 0
    # main program loop
    # exits when all cells have been searched
    while True:
        open_cells = []  # a list containing all coordinates that can be travelled to

        # cells that contain a value equal to the furthest distance from the start
        next_cells = get_cells_by_value(maze, start_dist)

        for cell in next_cells:
            neighbours = get_cell_neighbours(maze, cell)  # get all open neighbouring cells

            for neighbour in neighbours:
                if not neighbour in open_cells:
                    open_cells.append(neighbour)  # append all neighbours to our master open coords list

        if not open_cells:  # if there were no more open coordinates
            set_cell_value(maze, end_pos, start_dist + 1)
            break  # then we must have parsed every cell in the maze

        for cell in open_cells:
            if cell[0] > current_progress:
                current_progress = cell[0]

            set_cell_value(maze, cell, start_dist + 1)

        start_dist += 1

    final_path = get_final_path(maze, end_pos)
    next_cell = final_path[-2]
    action = 0  # left
    if next_cell[1] > start_pos[1]:
        action = 1  # right
    elif next_cell[0] < start_pos[0]:
        action = 2  # up
    elif next_cell[0] > start_pos[0]:
        action = 3  # down
    pi = np.zeros(4)
    pi[action] = 1
    return pi, -len(final_path) + 1


class Solver:

    def predict_one(self, state, three_d=False):
        return pi(state, three_d)
