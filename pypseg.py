import random
import collections
import logging

from pyseg import utils

# pypseg model

logging.basicConfig(level = logging.DEBUG, format = '[%(asctime)s] %(message)s',
                    datefmt = '%d/%m/%Y %H:%M:%S')


from pyseg.dpseg import Lexicon, State, Utterance

# Unigram case
class PYPState(State): # Information on the whole document
    def __init__(self, data, discount, alpha_1, p_boundary):
        # State parameters
        self.discount = discount
        utils.check_value_between(discount, 0, 1)
        self.alpha_1 = alpha_1
        self.p_boundary = p_boundary
        utils.check_probability(p_boundary)

        self.beta = 2 # Hyperparameter?

        logging.info(' discount: {0:.1f}, alpha_1: {1:d}, p_boundary: {2:.1f}'.format(self.discount, self.alpha_1, self.p_boundary))

        # Data and Utterance object
        self.unsegmented = utils.unsegmented(data) #datafile.unsegmented(data)
        self.unsegmented_list = utils.text_to_line(self.unsegmented, True) # Remove empty string
        #self.unsegmented_list = utils.delete_value_from_vector(self.unsegmented_list, '') # Remove empty string

        # Variable to store alphabet, utterance, and lexicon information
        self.utterances = [] #? # Stored utterances
        self.boundaries = [] # In case a boundary element is needed

        for unseg_line in self.unsegmented_list: # rewrite with correct variable names
            # do next_reference function
            utterance = PYPUtterance(unseg_line, p_boundary)
            self.utterances += [utterance]
            self.boundaries += [utterance.line_boundaries]

        self.n_utterances = len(self.utterances) # Number of utterances

        # Lexicon object (Counter)
        self.word_counts = Lexicon() # Word counter
        init_segmented_list = utils.text_to_line(self.get_segmented(), True) # Remove empty string
        #init_segmented_list = utils.delete_value_from_vector(init_segmented_list, '')
        self.word_counts.init_lexicon_text(init_segmented_list)

        # Alphabet (list of letters)
        self.alphabet = utils.delete_value_from_vector(list(set(self.unsegmented)), '\n')
        self.alphabet_size = len(self.alphabet)

        # Phonem probability (dictionary)
        self.phoneme_ps = dict()
        #self.init_probs() # How to initialise the boundaries (here: random)
        self.init_phoneme_probs()


    #def init_phoneme_probs(self):

    # Probabilities
    #def p_cont(self):

    def p_word(self, string):
        '''
        Computes the prior probability of a string of length n:
        p_word = p_boundary * (1 - p_boundary)^(n - 1)
                * \prod_1^n phoneme(string_i)
        No alpha_1 in this model.
        '''
        p = 1
        for letter in string:
            p = p * self.phoneme_ps[letter]
        p = p * ((1 - self.p_boundary) ** (len(string) - 1)) * self.p_boundary
        return p

    # Sampling
    #def sample(self, temp):

    #def get_segmented(self):


# Utterance in unigram case
class PYPUtterance(Utterance): # Information on one utterance of the document
    def __init__(self, sentence, p_segment):
        self.sentence = sentence # Unsegmented utterance # Char
        self.p_segment = p_segment
        utils.check_probability(p_segment)

        self.line_boundaries = [] # Test to store boundary existence
        self.init_boundary() #

    def init_boundary(self): # Random case only
        for i in range(len(self.sentence) - 1): # Unsure for the range
            rand_val = random.random()
            if rand_val < self.p_segment:
                self.line_boundaries += [True]
            else:
                self.line_boundaries += [False]
        self.line_boundaries += [True]

    def numer_base(self, word, state):
        if state.word_counts.lexicon[word] == 0: # If the word is not in the lexicon
            base = 0
        else: # The word is in the lexicon
            base = state.word_counts.lexicon[word] - state.discount
        base += ((state.discount * state.word_counts.n_types) + state.alpha_1) * state.p_word(word)
        #print('numer_base: ', base)
        #print('new element: ', (state.discount * state.word_counts.n_types) * state.p_word(word))
        return base

    #def left_word(self, i):

    #def right_word(self, i):

    #def centre_word(self, i):

    #def sample(self, state, temp):

    def sample_one(self, i, state, temp):
        lexicon = state.word_counts
        left = self.left_word(i)
        right = self.right_word(i)
        centre = self.centre_word(i)
        ### boundaries is the boundary for the utterance only here
        if self.line_boundaries[i]: # Boundary at the i-th position ('yes' case)
            #print('yes case')
            lexicon.remove_one(left) #lexicon[left] = lexicon[left] - 1
            lexicon.remove_one(right)
            #print(left, lexicon.lexicon[left], right, lexicon.lexicon[right])
        else: # No boundary at the i-th position ('no' case)
            #print('no case')
            lexicon.remove_one(centre)
            #print(centre, lexicon.lexicon[centre])

        denom = lexicon.n_tokens + state.alpha_1
        #print('denom: ', denom)
        yes = state.p_cont() * self.numer_base(left, state) \
        * (self.numer_base(right, state) + utils.kdelta(left, right)) / (denom + 1)
        #print('yes: ', yes)
        no = self.numer_base(centre, state)
        #print('no: ', no)

        # Normalisation
        yes = yes / (yes + no)
        no = 1 - yes

        # Annealing
        yes = yes ** temp
        #print('yes temp: ', yes)
        no = no ** temp
        p_yes = yes / (yes + no)
        if (random.random() < p_yes):
            #print('Boundary case')
            self.line_boundaries[i] = True
            lexicon.add_one(left)
            lexicon.add_one(right)
        else:
            #print('No boundary case')
            self.line_boundaries[i] = False
            lexicon.add_one(centre)

    #def prev_boundary(self, i):

    #def next_boundary(self, i):
