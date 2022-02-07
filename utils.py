import collections
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

def add_EOS(text_list):
    '''Add an EOS mark ($) at the end of each sentence.'''
    return [f'{sentence}$' for sentence in text_list]

# Checking functions
def check_probability(prob):
    '''Check that the given value is a probability.'''
    #if ((prob >= 0) and (prob <= 1)):
    #    pass
    #else:
    #    raise ValueError(f'{prob} must be between 0 and 1 to be a probability.')
    assert ((prob >= 0) and (prob <= 1)), (f'{prob} must be '
            'between 0 and 1 to be a probability.')

def check_value_between(value, lower_b, upper_b):
    '''Check that the given value is above a certain value and below another.'''
    #if (value >= lower_b) and (value < upper_b):
    #    pass
    #else:
    #    raise ValueError(f'{value} must be between {lower_b} and {upper_b}.')
    assert ((value >= lower_b) and (value < upper_b)), (f'{value} must be '
            f'between {lower_b} and {upper_b}.')

def check_equality(value_left, value_right):
    '''Check that both given values are equal.'''
    #if (value_left == value_right):
    #    pass
    #else:
    #    raise ValueError('Both values must be equal; '
    #                     f'currently {value_left} and {value_right}.')
    assert (value_left == value_right), ('Both values must be equal; '
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
    '''Kronecker delta: return 1 if i and j have the same value, 0 otherwise.'''
    return (i == j) # 0 or 1

def indicator(element, subset):
    '''Indicator function: return 1 if the element is in the subset, 0 otherwise.'''
    #if element in subset:
    #    return 1
    #else:
    #    return 0
    return element in subset # 0 or 1

# Useful functions
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

def bigram_list(word_list):
    '''Create a list of bigrams from a list of unigrams.'''
    return list(zip(word_list, word_list[1:]))

def morpheme_gold_segment(text, morpheme=False):
    '''Segment the gold text into word or morpheme level.'''
    if morpheme: # Morpheme level segmentation
        return re.sub('-', ' ', text)
    else: # Word level segmentation
        return re.sub('-', '', text)

def segment_sentence_with_boundaries(sentence, boundaries):
    '''Segment a sentence (string) according to a boundary vector (list).

    sentence is an unsegmented sentence.
    '''
    segmented_list = []
    beg = 0
    #pos = 0
    end = len(sentence)
    check_equality(end, len(boundaries))
    #for boundary in boundaries:
    #    if boundary: # If there is a boundary
    #        segmented_list += [sentence[beg:(pos + 1)]]
    #        beg = pos + 1
    #    pos += 1
    while beg < end:
        pos = boundaries.index(True, beg)
        segmented_list.append(sentence[beg:(pos + 1)])
        beg = pos + 1
    return segmented_list

def count_supervision_boundaries(sup_boundary_list):
    '''Count the number of supervision boundaries (and some statistics).'''
    flat_sup_boundaries = flatten_2D(sup_boundary_list)
    print('Number of boundaries:', len(flat_sup_boundaries))
    counter_sup_boundaries = collections.Counter(flat_sup_boundaries)
    print('Counter of boundaries:', counter_sup_boundaries)
    print('Ratio supervision boundary:', (counter_sup_boundaries[1] +
           counter_sup_boundaries[0]) / len(flat_sup_boundaries))
