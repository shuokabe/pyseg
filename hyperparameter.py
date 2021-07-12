import math
import random

#from numpy.random import default_rng
from numpy.random import RandomState
from scipy.stats import beta, gamma

class Lexicon: # Improved dictionary using a Counter
    '''Keep track of the lexicon in the dpseg model, using a Counter object.

    Attributes
    ----------
    lexicon : Counter
        Improved dictionary where each word is associated to its frequency.
        Format: {word_label: word frequency}
    n_types : integer
        Total number of types in the lexicon.
    n_tokens : int):
        Total number of tokens in the lexicon.

    '''
    def __init__(self):
        self.lexicon = collections.Counter()

        self.tables = dict()
        self.n_tables = 0

        # Number of types and tokens (size parameters)
        self.n_types = 0
        self.n_tokens = 0

        self.random_gen = random.Random(seed) # Avoid issues with main random numbers

    def update_lex_size(self):
        '''Updates the two parameters storing the number of tokens and of types.

        To use after any modification of the lexicon.
        '''
        self.n_types = len(self.lexicon)
        self.n_tokens = sum(self.lexicon.values())

    def add_one(self, word):
        '''
        Adds one count to a given word in the lexicon and updates the size
        parameters of the whole lexicon.
        '''
        self.lexicon[word] += 1

        if self.lexicon[word] == 1: # If it is a new word
            self.n_types += 1
        else: # If it was already in the dictionary
            pass
        #self.update_lex_size()
        self.n_tokens += 1 # Add one token

    def remove_one(self, word):
        '''
        Removes one count from a given word in the lexicon and updates the size
        parameters of the whole lexicon. Removes the word type from the lexicon
        if the count is 0.
        '''
        if self.lexicon[word] > 1:
            self.lexicon[word] += -1
        elif (self.lexicon[word] == 1): # If the count for the word is 0
            del self.lexicon[word]
            self.n_types += -1 # Remove the word from the lexicon
        else:
            raise KeyError('The word %s is not in the lexicon.' % word)
        #self.update_lex_size()
        self.n_tokens += -1 # Remove one token

    def init_lexicon_text(self, text):
        '''Initialises the lexicon (Counter) with the text.'''
        counter = collections.Counter()
        for line in text:
            split_line = utils.line_to_word(line)
            counter += collections.Counter(split_line) # Keep counter type
        self.lexicon = counter
        self.update_l


class Concentration_sampling:
    def __init__(self, prior, seed=42):
        '''Object to ease sampling the concentration parameter.

        Parameters
        ----------
        a_prior : float
            Gamma prior (shpae parameter) for the concentration parameter
        b_prior : float
            Gamma prior (scale parameter) for the concentration parameter
        seed : integer
            Seed value to generate random values (exclusive to this object)

        Attributes
        ----------
        a_prior : float
            Gamma prior (shpae parameter) for the concentration parameter
        b_prior : float
            Gamma prior (scale parameter) for the concentration parameter
        random_gen : default_rng()
            Random number generator for the scipy random variables

        '''
        self.a_prior, self.b_prior = prior
        #self.random_gen = default_rng(seed)
        self.random_gen = RandomState(seed) # Legacy Random Generation

    def sample_concentration(self, state):
        '''Sampling the concentration hyperparameter alpha.'''
        #n, k = state.word_counts.n_tokens, state.word_counts.n_types
        n, k = state.restaurant.n_customers, state.restaurant.n_tables
        # Auxiliary varaible eta
        eta = self.random_gen.beta(state.alpha_1 + 1, n)
        #eta = beta.rvs(state.alpha_1 + 1, n, random_state = self.random_gen)
        new_a = self.a_prior + k - 1 # Shape parameter
        new_b = self.b_prior - math.log(eta) # Scale parameter
        pi_eta = new_a / (new_a + (n * new_b))
        gamma_1 = self.random_gen.gamma(self.a_prior + k, new_b)
        #gamma_1 = gamma.rvs(self.a_prior + k, scale = new_b, random_state = self.random_gen)
        gamma_2 = self.random_gen.gamma(new_a, new_b)
        #gamma_2 = gamma.rvs(new_a, scale = new_b, random_state = self.random_gen)
        new_alpha = pi_eta * gamma_1 + (1 - pi_eta) * gamma_2
        return new_alpha
