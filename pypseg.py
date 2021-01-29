import collections
import logging
import random

from pyseg import utils

# pypseg model

logging.basicConfig(level = logging.DEBUG, format = '[%(asctime)s] %(message)s',
                    datefmt = '%d/%m/%Y %H:%M:%S')


from pyseg.dpseg import Lexicon, State, Utterance

# Exclusive to pypseg to count the number of tables
class Restaurant:
    def __init__(self, alpha_1, discount, seed=42):
        self.restaurant = dict() # word_label: list of number of customers (tables)
        self.tables = dict() # word_label: number of tables
        self.n_tables = 0 # sum(self.tables.values())
        self.customers = dict() # word_label: total number of customers for the word
        self.n_customers = 0 # sum(self.customers.values())

        # Parameters of the model
        self.alpha_1 = alpha_1
        self.discount = discount
        self.seed = seed
        self.random_gen = random.Random(self.seed) # Avoid issues with main random numbers

    def add_customer(self, word, random_value=None):
        '''Assign a customer (word) to a table in the restaurant.'''
        utils.check_equality(len(self.customers.keys()), len(self.restaurant.keys()))
        if word in self.restaurant.keys(): # Add the customer to a table (possibly new)
            n_customers = self.customers[word]
            #new_customer = random.randint(1, n_customers + self.alpha_1)
            #random_state = random.getstate() # Avoid issues with random numbers
            if random_value is not None:
                pass
            else:
                random_value = self.random_gen.random() #random.random()
            new_customer = random_value * (n_customers + self.alpha_1)
            #random.setstate(random_state)
            utils.check_equality(self.tables[word], len(self.restaurant[word]))
            if (new_customer > (n_customers - (self.discount * self.tables[word]))): # Open a new table
                self.restaurant[word].append(1)
                self.tables[word] += 1
                self.n_tables += 1
            else: # Add the new customer in an existing table
                cumulative_sum = 0
                for k in range(self.tables[word]):
                    cumulative_sum += (self.restaurant[word][k] - self.discount)
                    if new_customer <= cumulative_sum: # Add the customer to that table
                        self.restaurant[word][k] += 1
                        break
                    else:
                        pass
            self.customers[word] += 1
            self.n_customers += 1
            #utils.check_equality(self.customers[word], n_customers + 1)

        else: # Open a new table for a new word
            self.restaurant[word] = [1]
            self.tables[word] = 1
            self.n_tables += 1
            self.customers[word] = 1
            self.n_customers += 1

    def remove_customer(self, word):
        '''Remove a customer (word) from a table and close it if necessary.'''
        n_word_table = len(self.restaurant.get(word, []))
        if (n_word_table == 0):
            raise KeyError('There is no table with the word label %s.' % word)
        elif (n_word_table == 1): # Only one table
            self.restaurant[word][0] += -1
            self.customers[word] += -1
            self.n_customers += -1
            if (self.restaurant[word] == [0]): # Close the table
                del self.restaurant[word]
                del self.customers[word]
                del self.tables[word]
                self.n_tables += -1
        else: # More than one table
            n_customers = self.customers[word]
            #random_state = random.getstate() # Avoid issues with random numbers
            #new_customer = random.randint(1, n_customers)
            new_customer = self.random_gen.random() * n_customers #random.random() * n_customers
            #random.setstate(random_state)
            cumulative_sum = 0
            utils.check_equality(self.tables[word], len(self.restaurant[word]))
            for k in range(self.tables[word]):
                cumulative_sum += self.restaurant[word][k]
                if new_customer <= cumulative_sum: # Add the customer to that table
                    self.restaurant[word][k] += -1
                    self.customers[word] += -1
                    self.n_customers += -1
                    if (self.restaurant[word][k] == 0): # Close the table
                        del self.restaurant[word][k]
                        self.tables[word] += -1
                        self.n_tables += -1
                    break
                else:
                    pass
            #utils.check_equality(self.customers.get(word, n_customers - 1), n_customers - 1)

    def init_tables(self, text):
        '''Initialise the tables in the restaurant with the given text.'''
        for line in text:
            split_line = utils.line_to_word(line)
            for word in split_line:
                self.add_customer(word)


# Unigram case
class PYPState(State): # Information on the whole document
    def __init__(self, data, discount, alpha_1, p_boundary, seed=42):
        # State parameters
        self.discount = discount
        utils.check_value_between(discount, 0, 1)
        self.alpha_1 = alpha_1
        self.p_boundary = p_boundary
        utils.check_probability(p_boundary)

        self.beta = 2 # Hyperparameter?

        logging.info(' discount: {0:.1f}, alpha_1: {1:d}, p_boundary: {2:.1f}'.format(self.discount, self.alpha_1, self.p_boundary))

        self.seed = seed

        # Data and Utterance object
        self.unsegmented = utils.unsegmented(data) #datafile.unsegmented(data)
        self.unsegmented_list = utils.text_to_line(self.unsegmented, True) # Remove empty string

        # Variable to store alphabet, utterance, and lexicon information
        self.utterances = [] # Stored Utterance objects

        for unseg_line in self.unsegmented_list: # rewrite with correct variable names
            # do next_reference function
            utterance = PYPUtterance(unseg_line, self.p_boundary)
            self.utterances.append(utterance)

        self.n_utterances = len(self.utterances) # Number of utterances

        # Lexicon object (Counter)
        self.word_counts = Lexicon() # Word counter
        init_segmented_list = utils.text_to_line(self.get_segmented(), True) # Remove empty string
        self.word_counts.init_lexicon_text(init_segmented_list)

        # Tables object to count the number of tables (dict)
        self.restaurant = Restaurant(self.alpha_1, self.discount, self.seed)
        #random_state = random.getstate() # Avoid issues with random numbers
        self.restaurant.init_tables(init_segmented_list)
        #random.setstate(random_state)
        #print('Restaurant:', self.restaurant.restaurant)
        logging.debug('{} tables initially'.format(self.restaurant.n_tables))
        utils.check_value_between(self.restaurant.n_tables, self.word_counts.n_types, self.word_counts.n_tokens)
        utils.check_equality((sum(self.restaurant.customers.values())), self.word_counts.n_tokens)
        utils.check_equality(self.restaurant.n_customers, self.word_counts.n_tokens)

        # Alphabet (list of letters)
        self.alphabet = utils.delete_value_from_vector(list(set(self.unsegmented)), '\n')
        self.alphabet_size = len(self.alphabet)

        # Phoneme probability (dictionary)
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

        self.line_boundaries = []
        self.init_boundary()


    #def init_boundary(self): # Random case only

    def numer_base(self, word, state):
        #if state.word_counts.lexicon[word] == 0: # If the word is not in the lexicon
        #if word not in state.word_counts.lexicon: # If the word is not in the lexicon
        if word not in state.restaurant.customers: # If the word is not in the lexicon
            base = 0
        else: # The word is in the lexicon/restaurant
            #base = state.word_counts.lexicon[word] - (state.discount * state.restaurant.tables[word]) # state.discount
            base = state.restaurant.customers[word] - (state.discount * state.restaurant.tables[word])
        #base += ((state.discount * state.word_counts.n_types) + state.alpha_1) * state.p_word(word)
        base += ((state.discount * state.restaurant.n_tables) + state.alpha_1) * state.p_word(word)
        #print('numer_base: ', base)
        #print('new element: ', (state.discount * state.word_counts.n_types) * state.p_word(word))
        return base

    #def left_word(self, i):

    #def right_word(self, i):

    #def centre_word(self, i):

    #def sample(self, state, temp):

    def sample_one(self, i, state, temp):
        lexicon = state.word_counts
        restaurant = state.restaurant #
        left = self.left_word(i)
        right = self.right_word(i)
        centre = self.centre_word(i)
        ### boundaries is the boundary for the utterance only here
        #random_state = random.getstate() # Avoid issues with random numbers
        if self.line_boundaries[i]: # Boundary at the i-th position ('yes' case)
            #print('yes case')
            lexicon.remove_one(left) 
            lexicon.remove_one(right)
            restaurant.remove_customer(left) #
            restaurant.remove_customer(right) #
            #print(left, lexicon.lexicon[left], right, lexicon.lexicon[right])
        else: # No boundary at the i-th position ('no' case)
            #print('no case')
            lexicon.remove_one(centre)
            restaurant.remove_customer(centre) #
            #print(centre, lexicon.lexicon[centre])
        #random.setstate(random_state)

        denom = lexicon.n_tokens + state.alpha_1
        #denom = restaurant.n_customers + state.alpha_1
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
            lexicon.add_one(left)
            lexicon.add_one(right)
            restaurant.add_customer(left, random_value) #
            restaurant.add_customer(right, random_value) #
            utils.check_equality(restaurant.customers[left], lexicon.lexicon[left])
        else:
            #print('No boundary case')
            self.line_boundaries[i] = False
            lexicon.add_one(centre)
            restaurant.add_customer(centre, random_value) #
            utils.check_equality(restaurant.customers[centre], lexicon.lexicon[centre])

        utils.check_equality(restaurant.n_customers, lexicon.n_tokens)

    #def prev_boundary(self, i):

    #def next_boundary(self, i):
