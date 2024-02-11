############################################################
## CSC 384, Intro to AI, University of Toronto.
## Assignment 1 Starter Code
## v1.1
##
## Changes: 
## v1.1: removed the hfn paramete from dfs. Updated solve_puzzle() accordingly.
############################################################

from typing import List
import heapq
from heapq import heappush, heappop
import time
import argparse
import copy
import math # for infinity

from board import *

def is_goal(state : State): # Added data type here
    """
    Returns True if the state is the goal state and False otherwise.

    :param state: the current state.
    :type state: State
    :return: True or False
    :rtype: bool
    """

    # If all the boxes locations and storage locations match,
    # we are in the goal state
    if frozenset(state.board.boxes) == frozenset(state.board.storage):

        return True
    
    return False

    #raise NotImplementedError


def get_path(state : State): # Added data type here
    """
    Return a list of states containing the nodes on the path 
    from the initial state to the given state in order.

    :param state: The current state.
    :type state: State
    :return: The path.
    :rtype: List[State]
    """

    # Initializing path variable for returning
    path = []

    # Getting the path until we reach the root node
    while state.depth != 0:

        path.append(state)
        state = state.parent
    
    path.append(state)
    
    # Reversing the path for correct format and  then returning it
    path.reverse()
    return path

    #raise NotImplementedError


def get_successors(state : State): # Added data type here
    """
    Return a list containing the successor states of the given state.
    The states in the list may be in any arbitrary order.

    :param state: The current state.
    :type state: State
    :return: The list of successor states.
    :rtype: List[State]
    """

    # Initializing successor states list for returning
    succ_states = []

    # Setting up some variables for convenience
    init_robot_pos = state.board.robots[:]
    init_boxes_pos = state.board.boxes[:]
    obstacle_pos   = state.board.obstacles[:]

    # Going through every robot position
    for robot_pos in init_robot_pos:

        # Storing every possible robot position in a list
        new_robot = [(robot_pos[0] + 1, robot_pos[1]), (robot_pos[0] - 1, robot_pos[1]), 
                    (robot_pos[0], robot_pos[1] + 1), (robot_pos[0], robot_pos[1] - 1)]

        # Going through every possible new position of the robot as a new state
        for new_robot_pos in new_robot:

            # Checking if new robot position is an obstacle position or another robot position
            if (new_robot_pos not in obstacle_pos) and (new_robot_pos not in init_robot_pos) :

                # Calculating heuristic value of original board
                original_h_value = state.hfn(state.board)

                # Checking if the new robot position is a box position
                if new_robot_pos not in init_boxes_pos:

                    # Setting up the board for new successor state
                    new_board = Board(name= state.board.name[:], width= state.board.width, 
                                      height= state.board.height, robots= state.board.robots[:], 
                                      boxes= state.board.boxes[:], storage= state.board.storage[:], 
                                      obstacles= state.board.obstacles[:])
                    new_board.robots.remove(robot_pos)
                    new_board.robots.append(new_robot_pos)

                    # Setting up the new successor state
                    new_state = State(board= new_board, hfn= state.hfn, f= state.f + 1, 
                                      depth= state.depth + 1, parent= state)
                    new_state.f += state.hfn(new_board) - original_h_value

                    # Adding the new successor state to final list
                    succ_states.append(new_state)


                else:

                    # Trying to get direction in which the robot moved
                    x_diff = new_robot_pos[0] - robot_pos[0]
                    y_diff = new_robot_pos[1] - robot_pos[1]

                    # Adding the direction measured to get new position
                    new_box_pos = (new_robot_pos[0] + x_diff, new_robot_pos[1] + y_diff)

                    # Checking if new box position is valid
                    if ((new_box_pos not in init_boxes_pos) and (new_box_pos not in init_robot_pos) 
                        and (new_box_pos not in obstacle_pos)):

                        # Setting up the board for new successor state
                        new_board = Board(name= state.board.name[:], width= state.board.width, 
                                          height= state.board.height, robots= state.board.robots[:], 
                                          boxes= state.board.boxes[:], storage= state.board.storage[:], 
                                          obstacles= state.board.obstacles[:])
                        new_board.robots.remove(robot_pos)
                        new_board.robots.append(new_robot_pos)
                        new_board.boxes.remove(new_robot_pos)
                        new_board.boxes.append(new_box_pos)

                        # Setting up the new successor state
                        new_state = State(board= new_board, hfn= state.hfn, f= state.f + 1, 
                                          depth= state.depth + 1, parent= state)
                        new_state.f += state.hfn(new_board) - original_h_value

                        # Adding the new successor state to final list
                        succ_states.append(new_state)
    
    return succ_states
                        
    #raise NotImplementedError


def dfs(init_board):
    """
    Run the DFS algorithm given an initial board.

    If the function finds a goal state, it returns a list of states representing
    the path from the initial state to the goal state in order and the cost of
    the solution found.
    Otherwise, it returns am empty list and -1.

    :param init_board: The initial board.
    :type init_board: Board
    :return: (the path to goal state, solution cost)
    :rtype: List[State], int
    """

    # Setting up the initial state
    init_state = State(board= init_board, hfn= heuristic_zero, f= 0, depth= 0, parent= None)
    
    # Setting up the frontier, explored set and path variables
    frontier = [init_state]
    explored = set()
    path = []
    
    # Going through the DFS algorithm
    while len(frontier) != 0:
        
        # Getting the last added element from the frontier
        curr_state = frontier.pop()

        # Checking if curr_state is already explored or not
        if curr_state.id in explored:
            continue
        
        # Adding the curr_state to explored set
        explored.add(curr_state.id)
        
        # Checking if curr_state is the goal_state
        if is_goal(curr_state):

            # Find full path and cost of the path
            path = get_path(curr_state)
            cost = curr_state.f
            break
        
        else:
            
            # Get successor states and add them to the frontier
            succ_states = get_successors(curr_state)
            frontier.extend(succ_states)

    # Checking if a path was found
    if len(path) == 0:
        return path, -1
    
    # Return found path and its cost
    return path, cost

    #raise NotImplementedError


def a_star(init_board, hfn):
    """
    Run the A_star search algorithm given an initial board and a heuristic function.

    If the function finds a goal state, it returns a list of states representing
    the path from the initial state to the goal state in order and the cost of
    the solution found.
    Otherwise, it returns am empty list and -1.

    :param init_board: The initial starting board.
    :type init_board: Board
    :param hfn: The heuristic function.
    :type hfn: Heuristic (a function that consumes a Board and produces a numeric heuristic value)
    :return: (the path to goal state, solution cost)
    :rtype: List[State], int
    """
    # Setting up the initial state
    init_state = State(board= init_board, hfn= hfn, f= 0, depth= 0, parent= None)
    
    # Setting up the frontier, explored set and path variables
    frontier = [init_state]
    explored = set()
    path = []
    
    # Going through the A* algorithm
    while len(frontier) != 0:
        
        # Getting the last added element from the frontier
        curr_state = heappop(frontier)

        # Checking if curr_state is already explored or not
        if curr_state.id in explored:
            continue
        
        # Adding the curr_state to explored set
        explored.add(curr_state.id)
        
        # Checking if curr_state is the goal_state
        if is_goal(curr_state):

            # Find full path and cost of the path
            path = get_path(curr_state)
            cost = len(path) - 1
            break
        
        else:
            
            # Get successor states and add them to the frontier
            succ_states = get_successors(curr_state)
            for state in succ_states:
                heappush(frontier, state)

    # Checking if a path was found
    if len(path) == 0:
        return path, -1
    
    # Return found path and its cost
    return path, cost

    #raise NotImplementedError


def heuristic_basic(board : Board):
    """
    Returns the heuristic value for the given board
    based on the Manhattan Distance Heuristic function.

    Returns the sum of the Manhattan distances between each box 
    and its closest storage point.

    :param board: The current board.
    :type board: Board
    :return: The heuristic value.
    :rtype: int
    """

    # Making a copy of box positions and storage positions
    boxes_pos       = board.boxes[:]
    storage_pos     = board.storage[:]

    # Final heuristic value
    h_value = 0

    # Going through every storage position to find the closest box to it
    for box in boxes_pos:

        # Declaring a list for storing manhattan distances of storage points 
        # from the box location
        storage_distance = []

        # Going through each box
        for storage in storage_pos:

            # Calculating the manhattan distance and storing it
            manhattan = abs(box[0] - storage[0]) + abs(box[1] - storage[1])
            storage_distance.append(manhattan)
        
        # Finding the closest storage point to the box location
        h_value += min(storage_distance)

    # Returning heuristic value
    return h_value

    #raise NotImplementedError


def heuristic_advanced(board):
    """
    An advanced heuristic of your own choosing and invention.

    :param board: The current board.
    :type board: Board
    :return: The heuristic value.
    :rtype: int
    """

    # Making a copy of box positions and storage positions
    boxes_pos       = board.boxes[:]
    storage_pos     = board.storage[:]

    # Final heuristic value
    h_value = 0

    # Going through every storage position to find the closest box to it
    for box in boxes_pos:

        # Declaring a list for storing manhattan distances of storage points 
        # from the box location
        storage_distance = []
        
        # Checking if any box is in the corner and the corner position is not a storgae location
        if box not in storage_pos:
            if (box[0] + 1, box[1]) in board.obstacles and (box[0], box[1] + 1) in board.obstacles:
                h_value = math.inf
                break
            
            if (box[0] - 1, box[1]) in board.obstacles and (box[0], box[1] + 1) in board.obstacles:
                h_value = math.inf
                break
            
            if (box[0] + 1, box[1]) in board.obstacles and (box[0], box[1] - 1) in board.obstacles:
                h_value = math.inf
                break
            
            if (box[0] - 1, box[1]) in board.obstacles and (box[0], box[1] - 1) in board.obstacles:
                h_value = math.inf
                break

        # Going through each box
        for storage in storage_pos:

            # Calculating the manhattan distance and storing it
            manhattan = abs(box[0] - storage[0]) + abs(box[1] - storage[1])
            storage_distance.append(manhattan)
        
        # Finding the closest storage point to the box location
        h_value += min(storage_distance)
    
    # Returning heuristic value
    return h_value

    #raise NotImplementedError


def solve_puzzle(board: Board, algorithm: str, hfn):
    """
    Solve the given puzzle using the given type of algorithm.

    :param algorithm: the search algorithm
    :type algorithm: str
    :param hfn: The heuristic function
    :type hfn: Optional[Heuristic]

    :return: the path from the initial state to the goal state
    :rtype: List[State]
    """

    print("Initial board")
    board.display()

    time_start = time.time()

    if algorithm == 'a_star':
        print("Executing A* search")
        path, step = a_star(board, hfn)
    elif algorithm == 'dfs':
        print("Executing DFS")
        path, step = dfs(board)
    else:
        raise NotImplementedError

    time_end = time.time()
    time_elapsed = time_end - time_start

    if not path:

        print('No solution for this puzzle')
        return []

    else:

        print('Goal state found: ')
        path[-1].board.display()

        print('Solution is: ')

        counter = 0
        while counter < len(path):
            print(counter + 1)
            path[counter].board.display()
            print()
            counter += 1

        print('Solution cost: {}'.format(step))
        print('Time taken: {:.2f}s'.format(time_elapsed))

        return path


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="The file that contains the puzzle."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The file that contains the solution to the puzzle."
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=['a_star', 'dfs'],
        help="The searching algorithm."
    )
    parser.add_argument(
        "--heuristic",
        type=str,
        required=False,
        default=None,
        choices=['zero', 'basic', 'advanced'],
        help="The heuristic used for any heuristic search."
    )
    args = parser.parse_args()

    # set the heuristic function
    heuristic = heuristic_zero
    if args.heuristic == 'basic':
        heuristic = heuristic_basic
    elif args.heuristic == 'advanced':
        heuristic = heuristic_advanced

    # read the boards from the file
    board = read_from_file(args.inputfile)

    # solve the puzzles
    path = solve_puzzle(board, args.algorithm, heuristic)

    # save solution in output file
    outputfile = open(args.outputfile, "w")
    counter = 1
    for state in path:
        print(counter, file=outputfile)
        print(state.board, file=outputfile)
        counter += 1
    outputfile.close()