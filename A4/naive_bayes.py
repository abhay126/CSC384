############################################################
## CSC 384, Intro to AI, University of Toronto.
## Assignment 4 Starter Code
## v1.2
## - removed the example in ve since it is misleading.
## - updated the docstring in min_fill_ordering. The tie-breaking rule should
##   choose the variable that comes first in the provided list of factors.
############################################################

from bnetbase import Variable, Factor, BN
import csv


def normalize(factor):
    '''
    Normalize the factor such that its values sum to 1.
    Do not modify the input factor.

    :param factor: a Factor object. 
    :return: a new Factor object resulting from normalizing factor.
    '''

    # Getting the sum of all factor values
    total = sum(factor.values)

    # Getting the scope and name from given factor
    variables = factor.get_scope()
    name = 'Normalized.' + factor.name

    # Creating a new factor
    normalized_factor = Factor(name= name, scope= variables)
    
    # Normalizing original factor values and adding them to new factor
    for i in range(len(factor.values)):
        normalized_factor.values[i] = factor.values[i]/total

    return normalized_factor    

def restrict(factor, variable, value):
    '''
    Restrict a factor by assigning value to variable.
    Do not modify the input factor.

    :param factor: a Factor object.
    :param variable: the variable to restrict.
    :param value: the value to restrict the variable to
    :return: a new Factor object resulting from restricting variable to value.
             This new factor no longer has variable in it.
    ''' 

    # Getting the scope and name from given factor
    new_variables = factor.get_scope()
    new_variables.pop(new_variables.index(variable))

    name = 'Restricted.' + factor.name

    # Creating a new factor
    restricted_factor = Factor(name= name, scope= new_variables)

    # Getting the list of variables associated with original factor
    variables = factor.get_scope()

    # Creating a holder for indexing through values of new factor
    new_factor_index = 0

    # Going through every value of original factor
    for index in range(len(factor.values)):

        # Finding assigned indices of each variable based on the index value
        var_assignment_indices = variable_assignment_helper(factor= factor, index= index)

        # Going through every variable in the scope to find the restriced variable
        for i in range(len(variables)):

            # If variable is the restricted variable and variable assignment is the value to which 
            # it is restricted, put in values in the new factor
            if variables[i] == variable and variable.domain()[var_assignment_indices[i]] == value:
                restricted_factor.values[new_factor_index] = factor.values[index]
                new_factor_index += 1
                break
    
    return restricted_factor

def sum_out(factor, variable):
    '''
    Sum out a variable variable from factor factor.
    Do not modify the input factor.

    :param factor: a Factor object.
    :param variable: the variable to sum out.
    :return: a new Factor object resulting from summing out variable from the factor.
             This new factor no longer has variable in it.
    '''       
    # Getting the scope and name from given factor
    new_variables = factor.get_scope()
    sum_out_var_index = new_variables.index(variable)
    new_variables.pop(sum_out_var_index)

    name = 'Summed.' + factor.name

    # Creating a new factor
    summed_factor = Factor(name= name, scope= new_variables)

    # Initializing dictionary to hold all
    # possible combinations of assignment variable
    combinations = {}
    
    # Save original assignment indices of the factor variables
    saved_indices = []
    for var in factor.scope:
        saved_indices.append(var.get_assignment_index())

    for index in range(len(factor.values)):

        # Finding assigned indices of each variable based on the index value
        var_assignment_indices = variable_assignment_helper(factor= factor, index= index)
        
        # Set new indices for factor variables for easy access to associated values
        for i in range(len(factor.scope)):
            factor.scope[i].set_assignment_index(var_assignment_indices[i])

        # Remove the assignment index for summed out variable
        var_assignment_indices.pop(sum_out_var_index)
        
        # Populate the combinations dictionary based on the indices
        if tuple(var_assignment_indices) not in combinations:
            combinations[tuple(var_assignment_indices)] = factor.get_value_at_current_assignments()
        
        else:
            combinations[tuple(var_assignment_indices)] += factor.get_value_at_current_assignments()
    
    # Restore the assignment indices for the original factor variables
    for var in factor.scope:
        var.set_assignment_index(saved_indices[0])
        saved_indices[1:]
    
    # Save original assignment indices for the new factor variables
    saved_indices = []
    for var in summed_factor.scope:
        saved_indices.append(var.get_assignment_index())

    # Go through the combinations dictionary
    for combination, value in combinations.items():

        # Change assignment indicies according to dictionary key for easy value assignment in the factor
        for var in summed_factor.scope:
            var.set_assignment_index(combination[0])
            combination = combination[1:]
        
        # Assign the value to the factor
        summed_factor.add_value_at_current_assignment(value)
    
    # Restore the assignment indices for the new factor variables
    for var in summed_factor.scope:
        var.set_assignment_index(saved_indices[0])
        saved_indices[1:]
    
    return summed_factor
        
def multiply(factor_list):
    '''
    Multiply a list of factors together.
    Do not modify any of the input factors. 

    :param factor_list: a list of Factor objects.
    :return: a new Factor object resulting from multiplying all the factors in factor_list.
    ''' 
    # Checking how many factors are provided in the list
    if len(factor_list) < 2:

        # If there is only one factor in the factor list, return the only factor
        if len(factor_list) == 1:
            return factor_list[0]
    
        print("No factors provided")
        return
    
    else:
        
        # Initialize the final output factor
        multiplied_factor = None

        # Go through each factor within the factor list
        for factor_num in range(1, len(factor_list)):
            
            # Setup first factor to be multiplied
            f1 = multiplied_factor

            # If multiplied factor was none, assign f1 the previous factor in the list
            if not f1:
                f1 = factor_list[factor_num - 1]
            

            # Setup second factor to be multiplied
            f2 = factor_list[factor_num]

            # Get the scope of the new factor
            new_scope = []
            new_scope.extend(f1.get_scope())
            new_scope.extend(f2.get_scope())
            new_scope = list(set(new_scope))

            # Create a Factor object representing product of f1 and f2
            multiplied_factor = Factor(name= 'Mul(' + f1.name + ', ' + f2.name + ')', scope= new_scope)

            # Getting a list of common variables
            common_var = list(set(f1.get_scope()) & set(f2.get_scope()))

            # Going through all possible combinations of f1 and f2 values
            for index_1 in range(len(f1.values)):
                for index_2 in range(len(f2.values)):

                    # Setting valid flag to be true
                    valid = True

                    # Getting variable assignments based on factor values for both factors
                    var_assignment_indices_1 = variable_assignment_helper(f1, index_1)
                    var_assignment_indices_2 = variable_assignment_helper(f2, index_2)

                    # Checking if there are any common variables
                    if len(common_var):

                        # Checking if each common variable is assigned the same value
                        for variable in common_var:
                            if var_assignment_indices_1[f1.scope.index(variable)] != var_assignment_indices_2[f2.scope.index(variable)]:
                                
                                # Setting the valid flag to be false and breaking out of the loop
                                valid = False
                                break
                    
                    # Multiply the factor values only if it is valid to do so
                    if valid:

                        # Saving indices of all the variables
                        saved_indices = []
                        for variable in new_scope:
                            saved_indices.append(variable.get_assignment_index())
                        
                        # Setting assignments of all the variables in f1 based on f1 value under check
                        for i in range(len(f1.scope)):
                            f1.scope[i].set_assignment_index(var_assignment_indices_1[i])
                        
                        # Setting assignments of all the variables in f2 based on f2 value under check
                        for i in range(len(f2.scope)):
                            f2.scope[i].set_assignment_index(var_assignment_indices_2[i])
                        
                        # Get the associated multiplifaction factor value
                        multiplied_factor.add_value_at_current_assignment(f1.get_value_at_current_assignments() * 
                                                                          f2.get_value_at_current_assignments())

                        # Restore the assignment indices for the variables
                        for variable in new_scope:
                            variable.set_assignment_index(saved_indices[0])
                            saved_indices[1:]

        return multiplied_factor                        
                        



def min_fill_ordering(factor_list, variable_query):
    '''
    This function implements The Min Fill Heuristic. We will use this heuristic to determine the order 
    to eliminate the hidden variables. The Min Fill Heuristic says to eliminate next the variable that 
    creates the factor of the smallest size. If there is a tie, choose the variable that comes first 
    in the provided order of factors in factor_list.

    The returned list is determined iteratively.
    First, determine the size of the resulting factor when eliminating each variable from the factor_list.
    The size of the resulting factor is the number of variables in the factor.
    Then the first variable in the returned list should be the variable that results in the factor 
    of the smallest size. If there is a tie, choose the variable that comes first in the provided order of 
    factors in factor_list. 
    Then repeat the process above to determine the second, third, ... variable in the returned list.

    Here is an example.
    Consider our complete Holmes network. Suppose that we are given a list of factors for the variables 
    in this order: P(E), P(B), P(A|B, E), P(G|A), and P(W|A). Assume that our query variable is Earthquake. 
    Among the other variables, which one should we eliminate first based on the Min Fill Heuristic? 

    - Eliminating B creates a factor of 2 variables (A and E).
    - Eliminating A creates a factor of 4 variables (E, B, G and W).
    - Eliminating G creates a factor of 1 variable (A).
    - Eliminating W creates a factor of 1 variable (A).

    In this case, G and W tie for the best variable to be eliminated first since eliminating each variable 
    creates a factor of 1 variable only. Based on our tie-breaking rule, we should choose G since it comes 
    before W in the list of factors provided.
    '''

    # Initializing variable order list
    removing_order = []

    # Setting up initial list of all variables in the order they appear in factors
    all_variables = []
    for factor in factor_list:
        for variable in factor.scope:
            if variable != variable_query and variable not in all_variables:
                all_variables.append(variable)

    # Going through all the variables until the list is empty
    while len(all_variables) != 0:

        # Initializing the factor size list for each variable
        factor_size = []

        # Going through all the variables to determin factor sizes for each variable
        for variable in all_variables:

            # Setting up a list of all the factors that have current variable in their scope (relevant factors)
            rel_factors = []
            for factor in factor_list:
                if variable in factor.scope:
                    rel_factors.append(factor)
            
            # Getting the factor size based on relevant factors
            var_list = list(set([var for rel_factor in rel_factors for var in rel_factor.scope]))
            factor_size.append(len(var_list) - 1)
        
        # Getting the variable with smallest factor size
        var_index = factor_size.index(min(factor_size))
        removing_order.append(all_variables[var_index])

        # Popping the variable with smallest factor size from all_variables list
        all_variables.pop(var_index)
    
    return removing_order
    
    

def ve(bayes_net, var_query, varlist_evidence): 
    '''
    Execute the variable elimination algorithm on the Bayesian network bayes_net
    to compute a distribution over the values of var_query given the 
    evidence provided by varlist_evidence. 

    :param bayes_net: a BN object.
    :param var_query: the query variable. we want to compute a distribution
                     over the values of the query variable.
    :param varlist_evidence: the evidence variables. Each evidence variable has 
                         its evidence set to a value from its domain 
                         using set_evidence.
    :return: a Factor object representing a distribution over the values
             of var_query. that is a list of numbers, one for every value
             in var_query's domain. These numbers sum to 1. The i-th number
             is the probability that var_query is equal to its i-th value given 
             the settings of the evidence variables.

    '''

    # Getting all the factors from the bayesian network
    all_factors = bayes_net.factors()

    ### Step 1: Restrict the factors
    
    # Going through all evidence variables and factors 
    for evidence_var in varlist_evidence:
        for factor in all_factors:

            # If evidence variable is in factor's scope, restrict it and replace it in the original list
            if evidence_var in factor.scope:
                all_factors[all_factors.index(factor)] = restrict(factor= factor, 
                                                                  variable= evidence_var, 
                                                                  value= evidence_var.get_evidence())
    
    ### Step 2: Eliminate Hidden Variables

    # Getting the variable removal order based on heuristic
    variable_removal_order = min_fill_ordering(all_factors, var_query)

    # Going through each hidden variable
    while len(variable_removal_order) != 0:
        hidden_var = variable_removal_order.pop(0)
        
        # Getting a list of factors that contain the hidden variables (relevant factors)
        rel_factors = []
        for factor in all_factors:
            if hidden_var in factor.scope:
                rel_factors.append(factor)
        
        # Remove all relevant factors from original factors list
        for factor in rel_factors:
            all_factors.remove(factor)

        # Multiply all the relevant factors and sum out the hidden variable
        eliminating_factor = multiply(rel_factors)
        eliminating_factor = sum_out(eliminating_factor, hidden_var)

        # Add the resulting factor back to the original factors list
        all_factors.append(eliminating_factor)

    ### Step 3: Multiplying Remaining Factors
    final_factor = multiply(all_factors)
    
    ### Step 4: Normalize Factor
    final_factor = normalize(final_factor)

    return final_factor



## The order of these domains is consistent with the order of the columns in the data set.
salary_variable_domains = {
"Work": ['Not Working', 'Government', 'Private', 'Self-emp'],
"Education": ['<Gr12', 'HS-Graduate', 'Associate', 'Professional', 'Bachelors', 'Masters', 'Doctorate'],
"Occupation": ['Admin', 'Military', 'Manual Labour', 'Office Labour', 'Service', 'Professional'],
"MaritalStatus": ['Not-Married', 'Married', 'Separated', 'Widowed'],
"Relationship": ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
"Race": ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'],
"Gender": ['Male', 'Female'],
"Country": ['North-America', 'South-America', 'Europe', 'Asia', 'Middle-East', 'Carribean'],
"Salary": ['<50K', '>=50K']
}

salary_variable = Variable("Salary", ['<50K', '>=50K'])

def naive_bayes_model(data_file, variable_domains=salary_variable_domains, class_var=salary_variable):
    '''
    NaiveBayesModel returns a BN that is a Naive Bayes model that represents 
    the joint distribution of value assignments to variables in the given dataset.

    Remember a Naive Bayes model assumes P(X1, X2,.... XN, Class) can be represented as 
    P(X1|Class) * P(X2|Class) * .... * P(XN|Class) * P(Class).

    When you generated your Bayes Net, assume that the values in the SALARY column of 
    the dataset are the CLASS that we want to predict.

    Please name the factors as follows. If you don't follow these naming conventions, you will fail our tests.
    - The name of the Salary factor should be called "Salary" without the quotation marks.
    - The name of any other factor should be called "VariableName,Salary" without the quotation marks. 
      For example, the factor for Education should be called "Education,Salary".

    @return a BN that is a Naive Bayes model and which represents the given data set.
    '''
    ### READ IN THE DATA
    input_data = []
    with open(data_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None) #skip header row
        for row in reader:
            input_data.append(row)

    # Making variables out of variable_domains parameter for the bayesian network
    variable_list = []
    for variable_name, domain in variable_domains.items():
        variable_list.append(Variable(name= variable_name, domain= domain))
    
    # Getting the index of class variable in the variables list made
    class_var_index = None
    for i in range(len(variable_list)):
        if variable_list[i].name == class_var.name:
            class_var_index = i
            break
    
    # Getting the index of the string in headers that is class variable's name 
    class_headers_index = None
    for i in range(len(headers)):
        if headers[i] == class_var.name:
            class_headers_index = i
            break

    # Initializing factors list for the bayesian network
    factors = []

    # Making the factor associated with class variable
    class_factor = Factor(name= class_var.name, scope= [variable_list[class_var_index]])
    for data in input_data:
        class_val_index = class_var.domain().index(data[class_headers_index])
        class_factor.values[class_val_index] += 1
    
    # Making the factor associated with every other variable
    for i in range(len(variable_list)):
        if i != class_var_index:

            # Initializing the factor associated with variable
            factor_name = variable_list[i].name + ',' + variable_list[class_var_index].name
            factor_scope = [variable_list[i], variable_list[class_var_index]]
            factor = Factor(name= factor_name, scope= factor_scope)

            # Getting the index of variable data within the input data list
            for data_index in range(len(headers)):
                if headers[data_index] == variable_list[i].name:
                    break
            
            # Making a summary of counts for the variable
            data_summary = {}
            for data in input_data:
                
                # Increase the count of the variable assignments by 1
                if (data[data_index], data[class_headers_index]) not in data_summary:
                    data_summary[(data[data_index], data[class_headers_index])] = 1
                else:
                    data_summary[(data[data_index], data[class_headers_index])] += 1
            
            # Generating factor values to be added into the factor
            factor_values = []
            for assignments, counts in data_summary.items():
                class_val_index = class_var.domain().index(assignments[-1])
                value = list(assignments) + [counts/class_factor.values[class_val_index]]
                factor_values.append(value)
            
            # Add values to the factor and add the factor to the factors list
            factor.add_values(factor_values)
            factors.append(factor)

    # Normalizing the class factor and keeping its nam consistent
    class_factor = normalize(class_factor)
    class_factor.name = class_var.name
    factors.append(class_factor)
    
    # Making the Bayesian Network object
    bayes_net = BN(name= 'Naive Bayes Model', Vars= variable_list, Factors= factors)
    
    return bayes_net
            
        

def explore(bayes_net, question):
    '''    
    Return a probability given a Naive Bayes Model and a question number 1-6. 
    
    The questions are below: 
    1. What percentage of the women in the test data set does our model predict having a salary >= $50K? 
    2. What percentage of the men in the test data set does our model predict having a salary >= $50K? 
    3. What percentage of the women in the test data set satisfies the condition: P(S=">=$50K"|Evidence) is strictly greater than P(S=">=$50K"|Evidence,Gender)?
    4. What percentage of the men in the test data set satisfies the condition: P(S=">=$50K"|Evidence) is strictly greater than P(S=">=$50K"|Evidence,Gender)?
    5. What percentage of the women in the test data set with a predicted salary over $50K (P(Salary=">=$50K"|E) > 0.5) have an actual salary over $50K?
    6. What percentage of the men in the test data set with a predicted salary over $50K (P(Salary=">=$50K"|E) > 0.5) have an actual salary over $50K?

    @return a percentage (between 0 and 100)
    ''' 

    ### READ IN THE DATA
    input_data = []
    with open('data/adult-test.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None) #skip header row
        for row in reader:
            input_data.append(row)
    
    # Getting all the variables from bayesian network and initializing evidence variables list
    variables = bayes_net.variables()
    evidence_vars = []

    # Setting up the query variable
    for variable in variables:
        if variable.name == 'Salary':
            query_var = variable
            break

    # Checking which question we are answering
    if question == 1:

        # Initializing counters for rows and predictions
        female_counter = 0
        pred_counter = 0
        
        # Go through each data point
        for data in input_data:
            
            # Check if data point is for a woman
            if 'Female' in data:
                female_counter += 1

                # Set up evidence variables
                evidence_vars = set_evidence_helper(headers, data, variables, query_var)

                # Get the final factor and the prediction
                final_factor = ve(bayes_net= bayes_net, var_query= query_var, varlist_evidence= evidence_vars)
                if final_factor.get_value(['>=50K']) > 0.5:
                    pred_counter += 1
        
        return pred_counter/female_counter * 100

    elif question == 2:

        # Initializing counters for rows and predictions
        male_counter = 0
        pred_counter = 0
        
        # Go through each data point
        for data in input_data:
            
            # Check if data point is for a woman
            if 'Male' in data:
                male_counter += 1

                # Set up evidence variables
                evidence_vars = set_evidence_helper(headers, data, variables, query_var)

                # Get the final factor and the prediction
                final_factor = ve(bayes_net= bayes_net, var_query= query_var, varlist_evidence= evidence_vars)
                if final_factor.get_value(['>=50K']) > 0.5:
                    pred_counter += 1
        
        return pred_counter/male_counter * 100
    
    elif question == 3:

        # Initializing counters for rows and predictions
        female_counter = 0
        pred_counter = 0
        
        # Go through each data point
        for data in input_data:
            
            # Check if data point is for a woman
            if 'Female' in data:
                female_counter += 1

                # Set Gender variable's evidence value as Female
                for variable in variables:
                    if variable.name == 'Gender':
                        variable.set_evidence('Female')
                        gender_var = variable
                        break

                # Set up evidence variables
                evidence_vars = set_evidence_helper(headers, data, variables, query_var)

                # Get the final factor and the prediction
                final_factor = ve(bayes_net= bayes_net, var_query= query_var, varlist_evidence= evidence_vars)
                final_factor_gender = ve(bayes_net= bayes_net, var_query= query_var, varlist_evidence= evidence_vars + [gender_var])
                if final_factor.get_value(['>=50K']) > final_factor_gender.get_value(['>=50K']):
                    pred_counter += 1
        
        return pred_counter/female_counter * 100
    
    elif question == 4:

        # Initializing counters for rows and predictions
        male_counter = 0
        pred_counter = 0
        
        # Go through each data point
        for data in input_data:
            
            # Check if data point is for a woman
            if 'Male' in data:
                male_counter += 1

                # Set Gender variable's evidence value as Male
                for variable in variables:
                    if variable.name == 'Gender':
                        variable.set_evidence('Male')
                        gender_var = variable
                        break

                # Set up evidence variables
                evidence_vars = set_evidence_helper(headers, data, variables, query_var)

                # Get the final factor and the prediction
                final_factor = ve(bayes_net= bayes_net, var_query= query_var, varlist_evidence= evidence_vars)
                final_factor_gender = ve(bayes_net= bayes_net, var_query= query_var, varlist_evidence= evidence_vars + [gender_var])
                if final_factor.get_value(['>=50K']) > final_factor_gender.get_value(['>=50K']):
                    pred_counter += 1
        
        return pred_counter/male_counter * 100
    
    elif question == 5:

        # Initializing counters for rows and predictions
        high_salary_counter = 0
        pred_counter = 0
        
        # Go through each data point
        for data in input_data:
            
            # Check if data point is for a woman
            if 'Female' in data:

                # Set up evidence variables
                evidence_vars = set_evidence_helper(headers, data, variables, query_var)

                # Get the final factor and the prediction
                final_factor = ve(bayes_net= bayes_net, var_query= query_var, varlist_evidence= evidence_vars)
                if final_factor.get_value(['>=50K']) > 0.5:
                    pred_counter += 1

                    # Checking actual salary of the person
                    if '>=50K' in data:
                        high_salary_counter += 1
        
        return high_salary_counter/pred_counter * 100
    
    elif question == 6:

        # Initializing counters for rows and predictions
        high_salary_counter = 0
        pred_counter = 0
        
        # Go through each data point
        for data in input_data:
            
            # Check if data point is for a woman
            if 'Male' in data:

                # Set up evidence variables
                evidence_vars = set_evidence_helper(headers, data, variables, query_var)

                # Get the final factor and the prediction
                final_factor = ve(bayes_net= bayes_net, var_query= query_var, varlist_evidence= evidence_vars)
                if final_factor.get_value(['>=50K']) > 0.5:
                    pred_counter += 1

                    # Checking actual salary of the person
                    if '>=50K' in data:
                        high_salary_counter += 1
        
        return high_salary_counter/pred_counter * 100



def variable_assignment_helper(factor : Factor, index : int) -> list:
    """
    Takes in a factor and a valid index for its value and returns the indices of the
    assignments that were done for the variables of the factor in a list 

    :param factor: a Factor object.
    :param index: index of a factor value.
    :return: a List[int] containing the indices of values assigned to each variable
             from their domain.
    """

    # Get all the variables
    variables = factor.get_scope()
    variables.reverse()

    # Initializing output indices list
    variable_assignment_indices = []

    # Go through all the variables starting from the one at the end of the scope
    for variable in variables:

        # Find index of the assignment for each variable
        val_index = index % variable.domain_size()
        index = index // variable.domain_size()

        # Add the found assignment index to the output list
        variable_assignment_indices.append(val_index)
    
    variable_assignment_indices.reverse()

    return variable_assignment_indices

def set_evidence_helper(names : list, input_data : list, variables : list, 
                        query_var : Variable, evidence_names = ['Work', 'Education', 'Occupation', 'Relationship']) -> list:
    '''
    '''
    
    # Initializing evidence variables list
    evidence_vars = []
    
    # Going though all the names and variables
    for name in names:
        for variable in variables:

            # If the provided name is same as variable name and its not the query variable, set the evidence
            if variable.name == name and name not in query_var.name and name in evidence_names:
                variable.set_evidence(input_data[names.index(name)])
                evidence_vars.append(variable)
    
    return evidence_vars


if __name__ == "__main__":
    A = Variable(name= 'A', domain= [1,2,3])
    B = Variable(name= 'B', domain= ['a', 'b'])
    C = Variable(name= 'C', domain= ['heavy', 'light'])
    f1 = Factor(name= 'f1', scope= [A, B, C])
    f1.add_values([[1, 'a', 'heavy', 0.25], [1, 'a', 'light', 1.90],
                  [1, 'b', 'heavy', 0.50], [1, 'b', 'light', 0.80],
                  [2, 'a', 'heavy', 0.75], [2, 'a', 'light', 0.45],
                  [2, 'b', 'heavy', 0.99], [2, 'b', 'light', 2.25],
                  [3, 'a', 'heavy', 0.90], [3, 'a', 'light', 0.111],
                  [3, 'b', 'heavy', 0.01], [3, 'b', 'light', 0.1]])
    f1.print_table()

    print('#############      RESTRICT     ################')
    f2 = restrict(f1, C, 'heavy')
    f2.print_table()

    print("##############     SUM OUT         ###############")

    f3 = sum_out(f2, B)
    f3.print_table()

    print("##############    NORMALIZE    ################")
    f4 = normalize(f3)
    f4.print_table()

    print("##############     MULTIPLY     ################")
    f1 = Factor(name= 'f1', scope= [A, C])
    f2 = Factor(name= 'f2', scope= [B, C])
    f1.add_values([[1, 'heavy', 0.25],
                  [1, 'light', 0.80],
                  [2, 'heavy', 0.75],
                  [2, 'light', 2.25],
                  [3, 'heavy', 0.90],
                  [3, 'light', 0.01]])
    f2.add_values([['a', 'heavy', 0.25], ['a', 'light', 1.90],
                  ['b', 'heavy', 0.50], ['b', 'light', 0.80]])
    print("F1")
    f1.print_table()
    print("F2")
    f2.print_table()
    f3 = multiply([f1, f2])
    print("RESULT")
    f3.print_table()
    print("##########################################################")
    print(min_fill_ordering([f2, f1], C))
    print("##########################################################")
    bn = naive_bayes_model(data_file= 'adult-train.csv')
    for factor in bn.factors():
        factor.print_table()
    print("##########################################################")
    print(explore(bn, 1))
    print(explore(bn, 2))
    print(explore(bn, 3))
    print(explore(bn, 4))
    print(explore(bn, 5))
    print(explore(bn, 6))