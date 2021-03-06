import re

# Functions for text data
def unsegmented(text): # Raw string
    '''Return the unsegmented version of a raw text (string).'''
    return text.replace(' ', '')


# Splitting functions
def text_to_line(raw_text):
    r'''Split a raw text into a list of sentences (string) according to '\n'.'''
    split_text = re.split('\n', raw_text)
    if '' in split_text: # To remove empty lines
        return delete_value_from_vector(split_text, '')
    else:
        return split_text

def line_to_word(raw_line):
    '''Split a raw sentence into a list of words (string) according to whitespace.'''
    return re.split(' ', raw_line)


# Checking functions
def check_probability(prob):
    '''Check that the given value is a probability.'''
    if ((prob >= 0) and (prob <= 1)):
        pass
    else:
        raise ValueError(f'{prob} must be between 0 and 1 to be a probability.')

def check_value_between(value, lower_b, upper_b):
    '''Check that the given value is above a certain value and below another.'''
    if (value >= lower_b) and (value < upper_b):
        pass
    else:
        raise ValueError(f'{value} must be between {lower_b} and {upper_b}.')

def check_equality(value_left, value_right):
    '''Check that both given values are equal.'''
    if (value_left == value_right):
        pass
    else:
        raise ValueError('Both values must be equal; '
                         f'currently {value_left} and {value_right}.')

# Check the number of types and token
def check_n_type_token(state, args):
    '''Check the number of types and tokens after each sampling.'''
    if args.supervision_method in ['naive', 'naive_freq']:
        pass
    elif (args.model == 'pypseg') or (args.sample_hyperparameter): #model_name
        check_equality(state.restaurant.n_customers,
                       sum(state.restaurant.customers.values()))
        check_equality(state.restaurant.n_tables,
                       sum(state.restaurant.tables.values()))
    else:
        check_equality(state.word_counts.n_types,
                       len(state.word_counts.lexicon))
        check_equality(state.word_counts.n_tokens,
                       sum(state.word_counts.lexicon.values()))


# Other utility functions
def kdelta(i, j):
    '''Kronecker delta: return 1 if both i and j have the same value, 0 otherwise.'''
    if (i == j):
        return 1
    else:
        return 0

def indicator(element, subset):
    '''Indicator function: return 1 if the element is in the subset, 0 otherwise.'''
    if element in subset:
        return 1
    else:
        return 0

def delete_value_from_vector(vector, value):
    '''Delete a given value from a vector.

    To be used only when the value is in the vector.
    '''
    if value in vector:
        vector.remove(value)
        return vector
    else:
        raise ValueError('The asked value is not in the vector.')

def flatten_2D(list_of_list):
    '''Flatten a 2D list (list of list).'''
    return [element for element_list in list_of_list for element in element_list]
