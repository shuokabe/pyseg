import collections
import logging
import numpy as np
import random

from pyseg import utils

# Supervised pypseg model

logging.basicConfig(level = logging.DEBUG, format = '[%(asctime)s] %(message)s',
                    datefmt = '%d/%m/%Y %H:%M:%S')


from pyseg.model.pypseg import Restaurant, PYPState, PYPUtterance
from pyseg.model.supervised_dpseg import SupervisionHelper


# Exclusive to pypseg to count the number of tables
class Restaurant(Restaurant):
    #def __init__(self, alpha_1, discount, state, seed=42):

    #def phi(self, word):

    def add_naive_word(self, word, sup_parameter):
        '''Assign a word to a table in the restaurant for the naive method.

        A naive word from the supervision dictionary is not a 'real' customer.
        '''
        if word in self.restaurant.keys(): # Add the word to a table (possibly new)
            self.restaurant[word][0] += sup_parameter # Add to the first table
            self.customers[word] = self.customers.get(word, 0) + sup_parameter
        else: # Open a new table for a new word
            self.restaurant[word] = [sup_parameter]
            self.customers[word] = sup_parameter
            self.tables[word] = 1
            self.n_tables += 1

    def add_customer(self, word): #, random_value=None):
        '''Assign a customer (word) to a table in the restaurant.

        Modified version due to naive words not being customers.
        '''
        #utils.check_equality(len(self.customers.keys()), len(self.restaurant.keys()))
        if word in self.restaurant: #.keys(): # Add the customer to a table (possibly new)
            n_customers_w = self.customers.get(word, 0)
            if n_customers_w >= 1: # The word is currently in the text
                #n_customers = self.customers[word]
                n_tables_w = self.tables[word]
                random_value = self.random_gen.random()
                #new_customer = random_value * (n_customers_w + self.alpha_1)
                new_customer = random_value * self.phi(word)
                #* (n_customers_w + self.alpha_1 \
                   #+ self.discount * (self.n_tables - n_tables_w))
                utils.check_equality(self.tables[word], len(self.restaurant[word]))
                if (new_customer > (n_customers_w - (self.discount * n_tables_w))):
                    # Open a new table
                    self.open_table(word, False)
                else: # Add the new customer in an existing table
                    cumulative_sum = 0
                    for k in range(self.tables[word]):
                        cumulative_sum += (self.restaurant[word][k] - self.discount)
                        if new_customer <= cumulative_sum: # Add the customer to that table
                            self.restaurant[word][k] += 1
                            break
                        #else:
                        #    pass
                self.customers[word] += 1
            else: # Word from the naive dictionary and not in text
                # There is only one table with no real customer
                self.restaurant[word][0] += 1
                self.customers[word] = 1

        else: # Open a new table for a new word
            self.customers[word] = 1
            self.open_table(word, True)

        self.n_customers += 1

    def remove_customer(self, word):
        '''Remove a customer (word) from a table and close it if necessary.'''
        n_table_word = self.tables.get(word, [])
        if (n_table_word == 0):
            raise KeyError(f'There is no table with the word label {word}.')
        elif (n_table_word == 1): # Only one table
            self.restaurant[word][0] += -1
            self.customers[word] += -1
            if (self.restaurant[word][0] == 0): # Close the last table
                del self.customers[word]
                self.close_table(word, 0, True)
        else: # More than one table
            n_customers = self.customers[word]
            new_customer = self.random_gen.random() * n_customers
            cumulative_sum = 0
            utils.check_equality(self.tables[word], len(self.restaurant[word]))
            for k in range(self.tables[word]):
                cumulative_sum += self.restaurant[word][k]
                if new_customer <= cumulative_sum: # Add the customer to that table
                    self.restaurant[word][k] += -1
                    self.customers[word] += -1
                    if (self.restaurant[word][k] == 0): # Close the table
                        self.close_table(word, k, False)
                    elif (self.restaurant[word][k] < 1): # Naive word table
                        # Fuse table
                        self.restaurant[word][k + 1] += self.restaurant[word][k]
                        self.close_table(word, k, False)
                    break
                #else:
                #    pass
        self.n_customers += -1

    #def open_table(self, word, new_word=False):

    #def close_table(self, word, k, last_table=False):

    #def init_tables(self, text):


# Unigram case
class SupervisedPYPState(PYPState): # Information on the whole document
    def __init__(self, data, discount, alpha_1, p_boundary, seed=42,
                 #supervision_data=None, supervision_method='none',
                 #supervision_parameter=0, supervision_boundary='none',
                 #supervision_boundary_parameter=0):
                 supervision_helper=None):
        # State parameters
        self.discount = discount
        utils.check_value_between(discount, 0, 1)
        self.alpha_1 = alpha_1
        self.p_boundary = p_boundary
        utils.check_probability(p_boundary)

        self.beta = 2 # Hyperparameter?

        logging.info(f' discount: {self.discount:.1f}, alpha_1: '
                     f'{self.alpha_1:d}, p_boundary: {self.p_boundary:.1f}')

        self.seed = seed
        random_gen_sup = random.Random(self.seed)

        # Supervision helper
        self.sup = supervision_helper
        self.sup.supervision_logs()

        # Data and Utterance object
        self.unsegmented = utils.unsegmented(data)
        self.unsegmented_list = utils.text_to_line(self.unsegmented)

        # Variable to store alphabet, utterance, and lexicon information
        self.utterances = [] # Stored Utterance objects

        if self.sup.boundary_method != 'none': # Boundary supervision
            self.sup_boundaries = [] # Stored supervision boundaries
            sup_data_list = utils.text_to_line(data)
            len_unsegmented = len(self.unsegmented_list)
            utils.check_equality(len_unsegmented, len(sup_data_list))

            supervision_bool = False
            if self.sup.boundary_method == 'sentence':
                supervision_bool = True
                supervision_index = self.sup.supervision_index(len_unsegmented)
            for i in range(len_unsegmented):
                if (self.sup.boundary_method == 'sentence') \
                    and (i >= supervision_index): # End of supervision
                    supervision_bool = False
                unseg_line = self.unsegmented_list[i]
                sup_line = sup_data_list[i]
                if self.sup.boundary_method == 'word':
                    utterance = SupervisedPYPUtterance(
                        unseg_line, sup_line, self.p_boundary, random_gen_sup,
                        self.sup.boundary_method, self.sup.boundary_parameter,
                        sup_data = self.sup.data)
                else:
                    utterance = SupervisedPYPUtterance(
                        unseg_line, sup_line, self.p_boundary, random_gen_sup,
                        self.sup.boundary_method, self.sup.boundary_parameter,
                        supervision_bool)
                self.utterances.append(utterance)
                self.sup_boundaries.append(utterance.sup_boundaries)

            # Count number of supervision boundaries
            #print(self.sup_boundaries)
            flat_sup_boundaries = utils.flatten_2D(self.sup_boundaries)
            print('Number of boundaries:', len(flat_sup_boundaries))
            counter_sup_boundaries = collections.Counter(flat_sup_boundaries)
            print('Counter of boundaries:', counter_sup_boundaries)
            print('Ratio supervision boundary:', (counter_sup_boundaries[1] +
                   counter_sup_boundaries[0]) / len(flat_sup_boundaries))

        else: # Dictionary supervision (or no supervision) case
            for unseg_line in self.unsegmented_list:
                utterance = PYPUtterance(unseg_line, self.p_boundary)
                self.utterances.append(utterance)

        self.n_utterances = len(self.utterances) # Number of utterances
        if self.sup.boundary_method != 'none': # Boundary supervision
            utils.check_equality(len_unsegmented, self.n_utterances)

        init_segmented_list = utils.text_to_line(self.get_segmented())

        # Alphabet (list of letters)
        self.alphabet = utils.delete_value_from_vector(list(set(self.unsegmented)), '\n')
        self.alphabet_size = len(self.alphabet)

        # Phoneme probability (dictionary)
        self.phoneme_ps = dict()
        self.init_phoneme_probs()

        if self.sup.method in ['mixture', 'mixture_bigram']:
            # Total number of words in the supervision dictionary
            self.n_words_sup = sum(self.sup.data.values())

        # Restaurant object to count the number of tables (dict)
        self.restaurant = Restaurant(self.alpha_1, self.discount, self, self.seed)
        self.restaurant.init_tables(init_segmented_list)
        logging.debug(f'{self.restaurant.n_tables} tables initially')

        if self.sup.method == 'naive':
            for word, frequency in self.sup.data.items():
                self.restaurant.add_naive_word(word, self.sup.parameter)
            #print(f'{self.sup_method.capitalize()} restaurant:',
            #self.restaurant.restaurant)


    def init_phoneme_probs(self):
        '''
        Computes (uniform distribution)
        ### TODO: complete the documentation
        '''
        # Skip part to calculate the true distribution of characters

        if self.sup.method in ['init_bigram', 'mixture_bigram']:
            # Supervision with a dictionary
            #chosen_method = 'bigram'
            self.phoneme_ps = self.sup.set_bigram_character_model(self.alphabet)
            self.character_model = dict() # Dictionary to speed up the model

            print(f'Sum of probabilities: {sum(self.phoneme_ps.values())}')
        else:
            # Uniform distribution case
            logging.info('Phoneme distribution: uniform')

            for letter in self.alphabet:
                self.phoneme_ps[letter] = 1 / self.alphabet_size

    # Probabilities
    #def p_cont(self):

    def p_bigram_character_model(self, string):
        '''
        Probability from the bigram character model.
        It uses a dictionary as memory to store the values.
        '''
        if string in self.character_model:
            return self.character_model[string]
        else:
            p = 1
            n_ngram = 2 #
            considered_word = f'<{string:s}>'
            for i in range(len(considered_word) - n_ngram + 1):
                ngram = considered_word[i:(i + n_ngram)]
                p = p * self.phoneme_ps[ngram] #
            self.character_model[string] = p
            return p

    def p_word(self, string):
        '''
        Computes the prior probability of a string of length n:
        p_word = p_boundary * (1 - p_boundary)^(n - 1)
                * \prod_1^n phoneme(string_i)
        No alpha_1 in this model's function.
        '''
        p = 1
        # Character model
        if self.sup.method in ['init_bigram', 'init_trigram', 'mixture_bigram']:
            #if self.sup_method in ['init_bigram', 'mixture_bigram']:
            p = self.p_bigram_character_model(string)
        else: # Unigram case
            for letter in string:
                p = p * self.phoneme_ps[letter]
            if self.sup.method != 'initialise':
                p = p * ((1 - self.p_boundary) ** (len(string) - 1)) * self.p_boundary
        #p = p * ((1 - self.p_boundary) ** (len(string) - 1)) * self.p_boundary

        if self.sup.method in ['mixture', 'mixture_bigram']:
            #print('p before mixture:', p)
            p = (1 - self.sup.parameter) * p
            p += (self.sup.parameter / self.n_words_sup) \
                  * utils.indicator(string, self.sup.data)
            #print('p after mixture:', p)
        #elif self.sup_method == 'initialise': # Explicit length model
            #print('p before length:', p)
        #    p = p * self.word_length_ps.get(len(string), 10 ** (-5))
            #print('p after length:', p)
        #else:
        #    pass
        return p

    # Sampling
    #def sample(self, temp):

    #def get_segmented(self):


# Utterance in unigram case
class SupervisedPYPUtterance(PYPUtterance):
    '''Information on one utterance of the document'''
    def __init__(self, sentence, sup_sentence, p_segment, random_gen,
                 sup_boundary_method='none', sup_boundary_parameter=0,
                 supervision_bool=False, sup_data=dict()):
        self.sentence = sentence # Unsegmented utterance str
        self.sup_sentence = sup_sentence # Supervision sentence (with spaces)
        self.p_segment = p_segment
        utils.check_probability(p_segment)
        self.sentence_length = len(self.sentence)

        self.random_gen = random_gen

        self.line_boundaries = []
        self.init_boundary()

        self.sup_boundary_method = sup_boundary_method
        self.sup_boundary_parameter = sup_boundary_parameter

        self.sup_boundaries = []
        if (self.sup_boundary_method == 'sentence') and not supervision_bool:
            self.sup_boundaries = [-1] * (self.sentence_length) #len(self.sentence))
        elif (self.sup_boundary_method == 'word'):
            self.sup_data = sup_data
            self.init_word_sup_boundaries()
        else:
            self.init_sup_boundaries()

        utils.check_equality(self.sentence_length, len(self.sup_boundaries))


    #def init_boundary(self): # Random case only

    def init_sup_boundaries(self): # From SupervisedUtterance
        boundary_track = 0
        unseg_length = self.sentence_length #len(self.sentence)
        for i in range(unseg_length - 1):
            if self.sup_boundary_method == 'random':
                rand_val = self.random_gen.random()
                if rand_val >= self.sup_boundary_parameter:
                    self.sup_boundaries.append(-1)
                    if self.sup_sentence[boundary_track + 1] == ' ':
                        boundary_track += 1
                    boundary_track += 1
                    continue
            if self.sup_sentence[boundary_track + 1] == ' ': # Boundary case
                if self.sup_boundary_method == 'true':
                    rand_val = self.random_gen.random()
                    if rand_val > self.sup_boundary_parameter:
                        self.sup_boundaries.append(-1)
                        boundary_track += 2 #1
                        continue
                self.sup_boundaries.append(1)
                boundary_track += 1
            else: # No boundary case
                if self.sup_boundary_method in ['true', 'morpheme']:
                    self.sup_boundaries.append(-1)
                else:
                    self.sup_boundaries.append(0)
            boundary_track += 1
        self.sup_boundaries.append(1)

    def init_word_sup_boundaries(self):
        split_sup_sent = utils.line_to_word(self.sup_sentence)
        previous_word_known = True
        for word in split_sup_sent:
            word_length = len(word)
            if word in self.sup_data:
                if previous_word_known:
                    pass
                else:
                    self.sup_boundaries.pop()
                    self.sup_boundaries.append(1)
                self.sup_boundaries.extend([0] * (word_length - 1))
                self.sup_boundaries.append(1)
                previous_word_known = True
            else:
                self.sup_boundaries.extend([-1] * (word_length))
                previous_word_known = False
        self.sup_boundaries.pop()
        self.sup_boundaries.append(1)

    #def numer_base(self, word, state):

    #def left_word(self, i):

    #def right_word(self, i):

    #def centre_word(self, i):

    #def sample(self, state, temp):

    def sample_one(self, i, state, temp):
        #lexicon = state.word_counts
        restaurant = state.restaurant #
        left = self.left_word(i)
        right = self.right_word(i)
        centre = self.centre_word(i)
        ### boundaries is the boundary for the utterance only here

        if self.line_boundaries[i] == self.sup_boundaries[i]:
            return # No sampling if correct boundary status
        else:
            pass

        if self.line_boundaries[i]: # Boundary at the i-th position ('yes' case)
            #print('yes case')
            restaurant.remove_customer(left) #
            restaurant.remove_customer(right) #
            #print(left, lexicon.lexicon[left], right, lexicon.lexicon[right])
        else: # No boundary at the i-th position ('no' case)
            #print('no case')
            restaurant.remove_customer(centre) #
            #print(centre, lexicon.lexicon[centre])

        # Supervision
        if self.sup_boundaries[i] == 1:
            # No sampling if known boundary
            self.line_boundaries[i] = True
            restaurant.add_customer(left)
            restaurant.add_customer(right)
        elif self.sup_boundaries[i] == 0:
            # No sampling if known no boundary position
            self.line_boundaries[i] = False
            restaurant.add_customer(centre)
        else: # self.sup_boundaries[i] == -1: # Sampling case
            denom = restaurant.n_customers + state.alpha_1
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
            random_value = random.random()
            if (random_value < p_yes):
                #print('Boundary case')
                self.line_boundaries[i] = True
                restaurant.add_customer(left) #
                restaurant.add_customer(right) #
            else:
                #print('No boundary case')
                self.line_boundaries[i] = False
                restaurant.add_customer(centre) #

        if self.sup_boundaries[i] >= 0:
            utils.check_equality(self.sup_boundaries[i], self.line_boundaries[i])

    #def prev_boundary(self, i):

    #def next_boundary(self, i):
