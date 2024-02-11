############################################################
## CSC 384, Intro to AI, University of Toronto.
## Assignment 3 Starter Code
## v1.1
## Changes:
##   v1.1: updated the comments in kropki_model. 
##         the second return value should be a 2d list of variables.
############################################################

from board import *
from cspbase import *

def kropki_model(board):
    """
    Create a CSP for a Kropki Sudoku Puzzle given a board of dimension.

    If a variable has an initial value, its domain should only contain the initial value.
    Otherwise, the variable's domain should contain all possible values (1 to dimension).

    We will encode all the constraints as binary constraints.
    Each constraint is represented by a list of tuples, representing the values that
    satisfy this constraint. (This is the table representation taught in lecture.)

    Remember that a Kropki sudoku has the following constraints.
    - Row constraint: every two cells in a row must have different values.
    - Column constraint: every two cells in a column must have different values.
    - Cage constraint: every two cells in a 2x3 cage (for 6x6 puzzle) 
            or 3x3 cage (for 9x9 puzzle) must have different values.
    - Black dot constraints: one value is twice the other value.
    - White dot constraints: the two values are consecutive (differ by 1).

    Make sure that you return a 2D list of variables separately. 
    Once the CSP is solved, we will use this list of variables to populate the solved board.
    Take a look at csprun.py for the expected format of this 2D list.

    :returns: A CSP object and a list of variables.
    :rtype: CSP, List[List[Variable]]

    """
    # Initializing the CSP model
    kropki_csp = CSP(name= 'Kropki')
    
    # Setting up 2D list of variables
    board_var = [[0 for i in range(board.dimension)] for j in range(board.dimension)]

    # Creating variables with initial domains
    variables = create_variables(dim= board.dimension)

    # Adding each variable to the CSP
    for variable in variables:
        kropki_csp.add_var(variable)
    
    # Making row and column constraints
    row_col_cons = create_row_and_col_constraints(dim= board.dimension, 
                                                  sat_tuples= satisfying_tuples_difference_constraints(board.dimension), 
                                                  variables= variables)
    
    # Adding each constraint to the CSP
    for constraint in row_col_cons:
        kropki_csp.add_constraint(constraint)

    # Making same box constraints
    box_cons = create_cage_constraints(dim= board.dimension, 
                                       sat_tuples= satisfying_tuples_difference_constraints(board.dimension), 
                                       variables= variables)
    
    # Adding each constraint to the CSP
    for constraint in box_cons:
        kropki_csp.add_constraint(constraint)

    # Making dot constraints
    dot_cons = create_dot_constraints(dim= board.dimension, 
                                      dots= board.dots, 
                                      white_tuples= satisfying_tuples_white_dots(board.dimension), 
                                      black_tuples= satisfying_tuples_black_dots(board.dimension), 
                                      variables= variables)
    
    # Adding each constraint to the CSP
    for constraint in dot_cons:
        kropki_csp.add_constraint(constraint)

    # Filling the 2D list with variables and assign values to variables (if needed)
    for variable in variables:

        # Check whether the variable was assigned a value in the board
        if board.cells[int(variable.name[4])][int(variable.name[7])] != 0:

            # Assign a value to this variable
            variable.add_domain_values([board.cells[int(variable.name[4])][int(variable.name[7])]])
            variable.assign(board.cells[int(variable.name[4])][int(variable.name[7])])
        
        else:

            # Adding initial domains            
            variable.add_domain_values(create_initial_domain(board.dimension))

        # Add the variable to the 2D list
        board_var[int(variable.name[4])][int(variable.name[7])] = variable

    return kropki_csp, board_var
    # raise NotImplementedError
    
    
    
def create_initial_domain(dim):
    """
    Return a list of values for the initial domain of any unassigned variable.
    [1, 2, ..., dimension]

    :param dim: board dimension
    :type dim: int

    :returns: A list of values for the initial domain of any unassigned variable.
    :rtype: List[int]
    """

    # Creating the initial domain of an unassigned variable
    init_dom = [i + 1 for i in range(dim)]

    return init_dom
    #raise NotImplementedError



def create_variables(dim):
    """
    Return a list of variables for the board.

    We recommend that your name each variable Var(row, col).

    :param dim: Size of the board
    :type dim: int

    :returns: A list of variables, one for each cell on the board
    :rtype: List[Variables]
    """

    # Creating variable names
    var_names = ['Var({}, {})'.format(i, j) for i in range(dim) for j in range(dim)]

    # List of variables initialization
    variables = []

    # Making variables
    for name in var_names:
        var = Variable(name= name)
        variables.append(var)

    return variables
    #raise NotImplementedError

    
def satisfying_tuples_difference_constraints(dim):
    """
    Return a list of satifying tuples for binary difference constraints.

    :param dim: Size of the board
    :type dim: int

    :returns: A list of satifying tuples
    :rtype: List[(int,int)]
    """

    # Creating a list of satisfying tuples
    sat_tuples = [(i + 1, j + 1) for i in range(dim) for j in range(dim) if i != j]

    return sat_tuples
    # raise NotImplementedError
  
  
def satisfying_tuples_white_dots(dim):
    """
    Return a list of satifying tuples for white dot constraints.

    :param dim: Size of the board
    :type dim: int

    :returns: A list of satifying tuples
    :rtype: List[(int,int)]
    """

    # Creating a list of satisfying tuples
    sat_tuples = [(i + 1, j + 1) for i in range(dim) for j in range(dim) if abs(i - j) == 1]

    return sat_tuples
    # raise NotImplementedError
  
def satisfying_tuples_black_dots(dim):
    """
    Return a list of satifying tuples for black dot constraints.

    :param dim: Size of the board
    :type dim: int

    :returns: A list of satifying tuples
    :rtype: List[(int,int)]
    """

    # Creating a list of satisfying tuples
    sat_tuples = [(i + 1, j + 1) for i in range(dim) for j in range(dim) if (i + 1) == 2 * (j + 1) or 2 * (i + 1) == (j + 1)]

    return sat_tuples
    # raise NotImplementedError
    
def create_row_and_col_constraints(dim, sat_tuples, variables):
    """
    Create and return a list of binary all-different row/column constraints.

    :param dim: Size of the board
    :type dim: int

    :param sat_tuples: A list of domain value pairs (value1, value2) such that 
        the two values in each tuple are different.
    :type sat_tuples: List[(int, int)]

    :param variables: A list of all the variables in the CSP
    :type variables: List[Variable]
        
    :returns: A list of binary all-different constraints
    :rtype: List[Constraint]
    """
    # Initializing constraints list
    constraints = []

    # Initializing a list for keeping a check on adding same constraints with reversed scope
    checklist = []

    # Finding first variable for the constraint
    for var1 in variables:

        # Counter to make sure we are not searching more than necessary
        counter = 0

        # Finding second variable for the constraint
        for var2 in variables:

            # If the variable has same row or column as var1 (but is not var1)
            if ((int(var1.name[4]) == int(var2.name[4]) or int(var1.name[7]) == int(var2.name[7])) and 
                var1.name != var2.name and 
                (var1, var2) not in checklist and
                (var2, var1) not in checklist):
                
                # Add (var1, var2) and (var2, var1) in checklist
                checklist.extend([(var1, var2), (var2, var1)])

                # Increasing counter by 1
                counter += 1

                # Make a constraint and add satisfying tuples to it
                cons = Constraint(name= 'r_c({}, {})'.format(var1.name, var2.name), scope= [var1, var2])
                cons.add_satisfying_tuples(sat_tuples)

                # Add the constraint to the list
                constraints.append(cons)

                # Breaking this loop if counter has reached total number of possible constraints
                if counter == 2 * dim:
                    break
    
    return constraints
            
    #raise NotImplementedError
    
    
def create_cage_constraints(dim, sat_tuples, variables):
    """
    Create and return a list of binary all-different constraints for all cages.

    :param dim: Size of the board
    :type dim: int

    :param sat_tuples: A list of domain value pairs (value1, value2) such that 
        the two values in each tuple are different.
    :type sat_tuples: List[(int, int)]

    :param variables: A list of all the variables in the CSP
    :type variables: List[Variable]
        
    :returns: A list of binary all-different constraints
    :rtype: List[Constraint]
    """

    # Initializing constraints list
    constraints = []

    # Initializing a list for keeping a check on adding same constraints with reversed scope
    checklist = []

    # Finding first variable for the constraint
    for var1 in variables:

        # Finding second variable for the constraint
        for var2 in variables:
            
            # Initialziing row_size and col_size
            row_size, col_size = 3, 0

            # Checking the dimension
            if dim == 6:
                col_size = 2

            # Checking the dimension
            elif dim == 9:
                col_size = 3

            # Checking if the variables are in same block
            if (int(int(var1.name[4])/row_size) == int(int(var2.name[4])/row_size) and 
                int(int(var1.name[7])/col_size) == int(int(var2.name[7])/col_size) and
                var1.name != var2.name and
                (var1, var2) not in checklist and
                (var2, var1) not in checklist):

                # Add (var1, var2) and (var2, var1) in checklist
                checklist.extend([(var1, var2), (var2, var1)])

                # Make a constraint and add satisfying tuples to it
                cons = Constraint(name= 'b({}, {})'.format(var1.name, var2.name), scope= [var1, var2])
                cons.add_satisfying_tuples(sat_tuples)

                # Add the constraint to the list
                constraints.append(cons)     

    return constraints

    #raise NotImplementedError
    
def create_dot_constraints(dim, dots, white_tuples, black_tuples, variables):
    """
    Create and return a list of binary constraints, one for each dot.

    :param dim: Size of the board
    :type dim: int
    
    :param dots: A list of dots, each dot is a Dot object.
    :type dots: List[Dot]

    :param white_tuples: A list of domain value pairs (value1, value2) such that 
        the two values in each tuple satisfy the white dot constraint.
    :type white_tuples: List[(int, int)]
    
    :param black_tuples: A list of domain value pairs (value1, value2) such that 
        the two values in each tuple satisfy the black dot constraint.
    :type black_tuples: List[(int, int)]

    :param variables: A list of all the variables in the CSP
    :type variables: List[Variable]
        
    :returns: A list of binary dot constraints
    :rtype: List[Constraint]
    """

    # Initializing constraints list
    constraints = []

    # Going through each dot
    for dot in dots:

        # Getting position of dot
        if dot.location: # True means dot is between two cells in the same row

            # Setting up constraint variable initializations
            var1 = None
            var2 = None

            for variable in variables:
                if int(variable.name[4]) == dot.cell_row:
                    if int(variable.name[7]) == dot.cell_col:
                        var1 = variable
                    
                    elif int(variable.name[7]) == dot.cell_col + 1:
                        var2 = variable
            
            # Make constraints and add satisfying tuples to it
            cons = Constraint(name= 'd({}, {})'.format(var1.name, var2.name), scope= [var1, var2])

            # Checking the dot colour
            if dot.color == CHAR_BLACK:
                cons.add_satisfying_tuples(black_tuples)
            
            elif dot.color == CHAR_WHITE:
                cons.add_satisfying_tuples(white_tuples)
            
            # Add the constraints to the list
            constraints.append(cons)
        
        else:   # False means dot is between 2 cells in the same column

            # Setting up constraint variable initializations
            var1 = None
            var2 = None

            for variable in variables:
                if int(variable.name[7]) == dot.cell_col:
                    if int(variable.name[4]) == dot.cell_row:
                        var1 = variable
                    
                    elif int(variable.name[4]) == dot.cell_row + 1:
                        var2 = variable
            
            # Make constraints and add satisfying tuples to it
            cons = Constraint(name= 'd({}, {})'.format(var1.name, var2.name), scope= [var1, var2])

            # Checking the dot colour
            if dot.color == CHAR_BLACK:
                cons.add_satisfying_tuples(black_tuples)
            
            elif dot.color == CHAR_WHITE:
                cons.add_satisfying_tuples(white_tuples)

            # Add the constraints to the list
            constraints.append(cons)

    return constraints
    # raise NotImplementedError

