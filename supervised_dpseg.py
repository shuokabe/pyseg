import random
import collections
import logging

from pyseg import utils

# Supervised dpseg model

logging.basicConfig(level = logging.DEBUG, format = '[%(asctime)s] %(message)s',
                    datefmt = '%d/%m/%Y %H:%M:%S')

from pyseg.dpseg import Lexicon, State, Utterance


# Unigram case
class SupervisedState(State): # Information on the whole document
    def __init__(self, data, alpha_1, p_boundary, dictionary=None,
                 dictionary_method='none', dictionary_parameter=0):
        # State parameters
        self.alpha_1 = alpha_1
        self.p_boundary = p_boundary
        utils.check_probability(p_boundary)

        self.beta = 2 # Hyperparameter?

        logging.info(' alpha_1: {0:d}, p_boundary: {1:.1f}'.format(self.alpha_1, self.p_boundary))

        # Supervision variable
        self.dictionary = dictionary
        self.dictionary_method = dictionary_method
        self.dictionary_parameter = dictionary_parameter

        logging.info('Supervision with a dictionary')
        logging.info(' Supervision method: {0:s}, supervision parameter: {1:.2f}'.format(self.dictionary_method, self.dictionary_parameter))

        # Data and Utterance object
        self.unsegmented = utils.unsegmented(data) #datafile.unsegmented(data)
        self.unsegmented_list = utils.text_to_line(self.unsegmented, True) # Remove empty string
        #self.unsegmented_list = utils.delete_value_from_vector(self.unsegmented_list, '') # Remove empty string

        # Variable to store alphabet, utterance, and lexicon information
        self.utterances = [] #? # Stored utterances
        self.boundaries = [] # In case a boundary element is needed

        for unseg_line in self.unsegmented_list: # rewrite with correct variable names
            # do next_reference function
            utterance = Utterance(unseg_line, p_boundary)
            self.utterances += [utterance]
            self.boundaries += [utterance.line_boundaries]

        self.n_utterances = len(self.utterances) # Number of utterances

        # Lexicon object (Counter)
        self.word_counts = Lexicon() # Word counter
        init_segmented_list = utils.text_to_line(self.get_segmented(), True) # Remove empty string
        #init_segmented_list = utils.delete_value_from_vector(init_segmented_list, '')
        self.word_counts.init_lexicon_text(init_segmented_list)

        if self.dictionary_method == 'naive':
            naive_dictionary = {word: frequency * self.dictionary_parameter for word, frequency in self.dictionary.items()}
            self.word_counts.lexicon = self.word_counts.lexicon + collections.Counter(naive_dictionary)
            print('Naive dictionary', self.word_counts.lexicon)

        # Alphabet (list of letters)
        self.alphabet = utils.delete_value_from_vector(list(set(self.unsegmented)), '\n')
        self.alphabet_size = len(self.alphabet)

        # Phonem probability (dictionary)
        self.phoneme_ps = dict()
        #self.init_probs() # How to initialise the boundaries (here: random)
        self.init_phoneme_probs()


    def init_phoneme_probs(self):
        '''
        Computes (uniform distribution)
        ### to complete
        '''
        # Skip part to calculate the true distribution of characters

        #if self.dictionary:
        if self.dictionary_method == 'initialise':
            # Supervision with a dictionary
            logging.info('Phoneme distribution: dictionary supervision')

            #if self.dictionary_method == 'initialise':

            words_in_dict_str = ''
            for word, frequency in self.dictionary.items():
                words_in_dict_str += word * frequency
            print(words_in_dict_str[0:50])
            letters_in_dict = collections.Counter(words_in_dict_str)
            frequency_letters_dict = sum(letters_in_dict.values())

            for letter in self.alphabet:
                # TODO: deal with the case letters_in_dict[letter] == 0
                self.phoneme_ps[letter] = letters_in_dict[letter] / frequency_letters_dict
            #assert (abs(sum(self.phoneme_ps.values()) - 1.0) < 10^(-5)), 'The sum of the probabilities is not 1.'
            print('Sum of probabilities: {0}'.format(sum(self.phoneme_ps.values())))

        else:
            # Uniform distribution case
            logging.info('Phoneme distribution: uniform')

            for letter in self.alphabet:
                self.phoneme_ps[letter] = 1 / self.alphabet_size

    # Probabilities
    #def p_cont(self):

    def p_word(self, string):
        '''
        Computes the prior probability of a string of length n:
        p_word = alpha_1 * p_boundary * (1 - p_boundary)^(n - 1)
                * \prod_1^n phoneme(string_i)
        '''
        p = 1
        for letter in string:
            p = p * self.phoneme_ps[letter]
        p = p * ((1 - self.p_boundary) ** (len(string) - 1)) * self.p_boundary
        if self.dictionary_method == 'mixture':
            #print('p before mixture:', p)
            n_words_dict = sum(self.dictionary.values())
            p = self.dictionary_parameter / n_words_dict * utils.indicator(string, self.dictionary) \
                + (1 - self.dictionary_parameter) * p
            #print('p after mixture:', p)
        return p * self.alpha_1

    # Sampling
    #def sample(self, temp):

    #def get_segmented(self):

#class Utterance:
