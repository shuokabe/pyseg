import re


# Splitting functions
def text_to_line(raw_text):
    '''
    Splits a raw text into a list of sentences (string) according to '\n'.
    '''
    return re.split('\n', raw_text)

def line_to_word(raw_line):
    '''
    Splits a raw sentence into a list of words (string) according to ' '.
    '''
    return re.split(' ', raw_line)


# Checking functions
def check_probability(prob):
    '''
    Checks that the given value is a probability.
    '''
    if ((prob >= 0) and (prob <= 1)):
        pass
    else:
        raise ValueError('Value must be between 0 and 1 to be a probability.')

def check_value_between(value, lower_b, upper_b):
    '''
    Checks that the given value is above a certain value and below another.
    '''
    if (value >= lower_b) and (value < upper_b):
        pass
    else:
        raise ValueError('Value must be between %d and %d.' % (lower_b, upper_b))

def check_equality(value_left, value_right):
    '''
    Checks that both given values are equal.
    '''
    if (value_left == value_right):
        pass
    else:
        raise ValueError('Both values must be equal; currently %d and %d.' % (value_left, value_right))


# Other utility functions
def kdelta(i, j):
    '''
    Kronecker delta: returns 1 if both i and j have the same value, 0 otherwise.
    '''
    if (i == j):
        return 1
    else:
        return 0

def delete_value_from_vector(vector, value):
    '''
    Deletes a given value from a vector.
    To be used only when the value is in the vector.
    '''
    if value in vector:
        vector.remove(value)
        return vector
    else:
        raise ValueError('The asked value is not in the vector.')
