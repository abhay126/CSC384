############################################################
## CSC 384, Intro to AI, University of Toronto.
## Assignment 3 Starter Code
## v1.0
##
############################################################


def prop_FC(csp, last_assigned_var=None):
    """
    This is a propagator to perform forward checking. 

    First, collect all the relevant constraints.
    If the last assigned variable is None, then no variable has been assigned 
    and we are performing propagation before search starts.
    In this case, we will check all the constraints.
    Otherwise, we will only check constraints involving the last assigned variable.

    Among all the relevant constraints, focus on the constraints with one unassigned variable. 
    Consider every value in the unassigned variable's domain, if the value violates 
    any constraint, prune the value. 

    :param csp: The CSP problem
    :type csp: CSP
        
    :param last_assigned_var: The last variable assigned before propagation.
        None if no variable has been assigned yet (that is, we are performing 
        propagation before search starts).
    :type last_assigned_var: Variable

    :returns: The boolean indicates whether forward checking is successful.
        The boolean is False if at least one domain becomes empty after forward checking.
        The boolean is True otherwise.
        Also returns a list of variable and value pairs pruned. 
    :rtype: boolean, List[(Variable, Value)]
    """

    # Initialization of relvant constraints list
    rel_constraints = []

    # Initialization of variable-pruned value pair
    pruned_history = []

    # Check if there is a last assigned variable
    if last_assigned_var:

            # Get all constraints that have last_assigned_var in them
            rel_constraints = csp.get_cons_with_var(last_assigned_var)
        
    else:

        # All constraints are relevant otherwise
        rel_constraints = csp.get_all_cons()

    # Go through each relevant constraint
    for constraint in rel_constraints:

        # Check if the constraint has only one unassigned variable
        if constraint.get_num_unassigned_vars() == 1:

            # Get unassigned variables and their domains
            unassigned_var = constraint.get_unassigned_vars()[0]


            test_values = [variable.get_assigned_value() for variable in constraint.scope]

            # Go through every value in variable's current domain
            for value in unassigned_var.cur_domain():

                # Add the value in test_values
                test_values[constraint.scope.index(unassigned_var)] = value

                # Check whether the test_values are valid
                if not constraint.check(test_values):

                    # prune the value
                    unassigned_var.prune_value(value)

                    # Adding the variable and pruned value pair into pruned history
                    pruned_history.append((unassigned_var, value))

                    # Check if this unassigned variable has zero domain
                    if unassigned_var.cur_domain_size == 0:

                        return False, pruned_history
    
    return True, pruned_history

    # raise NotImplementedError


def prop_AC3(csp, last_assigned_var=None):
    """
    This is a propagator to perform the AC-3 algorithm.

    Keep track of all the constraints in a queue (list). 
    If the last_assigned_var is not None, then we only need to 
    consider constraints that involve the last assigned variable.

    For each constraint, consider every variable in the constraint and 
    every value in the variable's domain.
    For each variable and value pair, prune it if it is not part of 
    a satisfying assignment for the constraint. 
    Finally, if we have pruned any value for a variable,
    add other constraints involving the variable back into the queue.

    :param csp: The CSP problem
    :type csp: CSP
        
    :param last_assigned_var: The last variable assigned before propagation.
        None if no variable has been assigned yet (that is, we are performing 
        propagation before search starts).
    :type last_assigned_var: Variable

    :returns: a boolean indicating if the current assignment satisifes 
        all the constraints and a list of variable and value pairs pruned. 
    :rtype: boolean, List[(Variable, Value)]
    """
    # Initialization of relvant constraints list
    rel_constraints = []

    # Initialization of variable-pruned value pair
    pruned_history = []

    # Check if there is a last assigned variable
    if last_assigned_var:

            # Get all constraints that have last_assigned_var in them
            rel_constraints = csp.get_cons_with_var(last_assigned_var)
        
    else:

        # All constraints are relevant otherwise
        rel_constraints = csp.get_all_cons()
    
    # Go through each relevant constraint
    while len(rel_constraints) != 0:
       
        # Getting the first element of the constraint queue
        constraint = rel_constraints.pop(0)

        # Going through each variable in the constraint
        for variable in constraint.scope:

            # Checking if variable is assigned or not
            if not variable.is_assigned():

                # Going through each value in the variable's current domain
                for value in variable.cur_domain():

                    # Check for whether the value is valid is not
                    valid = False

                    # initializing a list to hold variable-value pair satisfactory tuples
                    pair_tuples = []

                    # Getting tuples associated with this variable-value pair
                    if (variable, value) in constraint.sup_tuples:
                        pair_tuples = constraint.sup_tuples[(variable, value)]

                    # Going through each tuple in the collected pair_tuples
                    for tup in pair_tuples:

                        # Variable to hold breaking flag
                        broke_out = False

                        # Checking if every element of the tuple is in its respective variable's domain
                        for i in range(len(tup)):
                            if not constraint.scope[i].in_cur_domain(tup[i]):
                                broke_out = True
                                break
                        
                        # If all the values of the tuple existed in respective variable's domain, the value under testing is valid
                        if not broke_out:
                            valid = True
                            break

                    # If the value is not valid
                    if not valid:
                        
                        # prune the value
                        variable.prune_value(value)

                        # Adding the variable and pruned value pair into pruned history
                        pruned_history.append((variable, value))

                        # Check if the current domain of variable is empty or not
                        if variable.cur_domain_size == 0:
                        
                            return False, pruned_history
                        
                        # Adding constraints back to the list
                        repeated_cons = csp.get_cons_with_var(variable)

                        # Going through each constraint that might be added back
                        for cons in repeated_cons:

                            # Making sure to not add a constraint that is already in the queue (or the same constraint)
                            if cons not in rel_constraints and cons != constraint:
                                rel_constraints.append(cons)

    return True, pruned_history
    
    # raise NotImplementedError

def ord_mrv(csp):
    """
    Implement the Minimum Remaining Values (MRV) heuristic.
    Choose the next variable to assign based on MRV.

    If there is a tie, we will choose the first variable. 

    :param csp: A CSP problem
    :type csp: CSP

    :returns: the next variable to assign based on MRV

    """

    # Initialize the next variable as the first variable in csp
    next_var = None

    # Go through each variable
    for variable in csp.get_all_unasgn_vars():

        # Check if this is the first initialization of next_var
        if not next_var:
            next_var = variable

        # Check if the variable has less number of remaining values
        elif variable.cur_domain_size() < next_var.cur_domain_size():

            # Assign the variable to next_var
            next_var = variable
    
    return next_var
    # raise NotImplementedError


###############################################################################
# Do not modify the prop_BT function below
###############################################################################


def prop_BT(csp, last_assigned_var=None):
    """
    This is a basic propagator for plain backtracking search.

    Check if the current assignment satisfies all the constraints.
    Note that we only need to check all the fully instantiated constraints 
    that contain the last assigned variable.
    
    :param csp: The CSP problem
    :type csp: CSP

    :param last_assigned_var: The last variable assigned before propagation.
        None if no variable has been assigned yet (that is, we are performing 
        propagation before search starts).
    :type last_assigned_var: Variable

    :returns: a boolean indicating if the current assignment satisifes all the constraints 
        and a list of variable and value pairs pruned. 
    :rtype: boolean, List[(Variable, Value)]

    """
    
    # If we haven't assigned any variable yet, return true.
    if not last_assigned_var:
        return True, []
        
    # Check all the constraints that contain the last assigned variable.
    for c in csp.get_cons_with_var(last_assigned_var):

        # All the variables in the constraint have been assigned.
        if c.get_num_unassigned_vars() == 0:

            # get the variables
            vars = c.get_scope() 

            # get the list of values
            vals = []
            for var in vars: #
                vals.append(var.get_assigned_value())

            # check if the constraint is satisfied
            if not c.check(vals): 
                return False, []

    return True, []
