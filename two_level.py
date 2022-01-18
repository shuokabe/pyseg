import collections
import logging
import random

# Two-level model (word and morpheme)

logging.basicConfig(level = logging.DEBUG, format = '[%(asctime)s] %(message)s',
                    datefmt = '%d/%m/%Y %H:%M:%S')

from pyseg.dpseg import State
from pyseg.pypseg import Restaurant, PYPState, PYPUtterance
from pyseg.supervised_dpseg import SupervisionHelper, SupervisedState
from pyseg.supervised_pypseg import SupervisedPYPState
from pyseg.hyperparameter import Hyperparameter_sampling
from pyseg import utils


class TwoLevelState(State):
    def __init__(self, raw_data, discount, alpha_1, p_boundary, seed=42,
                 supervision_helper=None):
        # raw_data is a text segmented in morphemes with '-'
        self.data_word = utils.morpheme_gold_segment(raw_data, False) # Word level
        self.data_morph = utils.morpheme_gold_segment(raw_data, True) # Morpheme level

        # Two-level sampling order
        self.word_then_morph = False
        logging.info('Two-level model')
        if self.word_then_morph:
            logging.info(' Sampling words and then morphemes')
        else:
            logging.info(' Sampling morphemes and then words')

        self.sup = supervision_helper
        # Two states for two levels
        if self.sup == None:
            print('Without supervision')
            logging.info('Word level model:')
            self.word_state = PYPState(self.data_word, discount = discount,
                                    alpha_1 = alpha_1, p_boundary = p_boundary,
                                    seed = seed)
            logging.info('Morpheme level model:')
            self.morph_state = PYPState(self.data_morph, discount = discount,
                                        alpha_1 = alpha_1, p_boundary = p_boundary,
                                        seed = seed)
        else:
            print('With supervision')
            # Prepare the two supervision helpers
            if self.sup.data == 'none':
                self.word_sup_data, self.morph_sup_data = 'none', 'none'
            else:
                self.set_two_level_supervision_dictionary()
            word_supervision_helper = SupervisionHelper(self.word_sup_data,
                self.sup.method, self.sup.parameter, self.sup.boundary_method,
                self.sup.boundary_parameter, verbose = False)
            morph_supervision_helper = SupervisionHelper(self.morph_sup_data,
                self.sup.method, self.sup.parameter, self.sup.boundary_method,
                self.sup.boundary_parameter, verbose = False)
            logging.info('Word level model:')
            self.word_state = SupervisedPYPState(self.data_word,
                                    discount = discount, alpha_1 = alpha_1,
                                    p_boundary = p_boundary, seed = seed,
                                    supervision_helper = word_supervision_helper)
            logging.info('Morpheme level model:')
            self.morph_state = SupervisedPYPState(self.data_morph,
                                    discount = discount, alpha_1 = alpha_1,
                                    p_boundary = p_boundary, seed = seed,
                                    supervision_helper = morph_supervision_helper)
        ### Harmonise the initial state

        self.n_utterances = len(self.word_state.utterances)
        utils.check_equality(len(self.morph_state.utterances), self.n_utterances)

        self.alphabet_size = self.word_state.alphabet_size # For main.py

        logging.info(' Hyperparameter sampled after each iteration.')
        self.word_hyper_sample = Hyperparameter_sampling((1, 1), (1, 1), seed, True)
        self.morph_hyper_sample = Hyperparameter_sampling((1, 1), (1, 1), seed, True)


    def set_two_level_supervision_dictionary(self):
        '''Create two supervision dictionaries for the corresponding levels.'''
        raw_sup_word = ' '.join(list(self.sup.data.keys()))
        print('gold sup dict', self.sup.data, len(self.sup.data))
        sup_word_list = utils.line_to_word(
                            utils.morpheme_gold_segment(raw_sup_word, False))
        #print('sup word list', sup_word_list)
        self.word_sup_data = collections.Counter(sup_word_list)
        #print('word dict', self.word_sup_data, len(self.word_sup_data.keys()))
        sup_morph_list = utils.line_to_word(
                            utils.morpheme_gold_segment(raw_sup_word, True))
        #print('sup morph list', sup_morph_list)
        self.morph_sup_data = collections.Counter(sup_morph_list)
        #print('morph dict', self.morph_sup_data, len(self.morph_sup_data.keys()))

    #def init_phoneme_probs(self):

    #def p_cont(self):

    #def p_word(self, string):

    def put_boundary(self, i, state, utterance):
        '''Put a boundary (1) at position i.

        This implies to remove and then add the adequate words.
        '''
        if utterance.line_boundaries[i]: # Boundary at the i-th position ('yes' case)
            pass
            #lexicon.remove_one(left)
            #lexicon.remove_one(right)
        else: # No boundary at the i-th position ('no' case)
            #lexicon = state.word_counts
            restaurant = state.restaurant #
            left = utterance.left_word(i)
            right = utterance.right_word(i)
            centre = utterance.centre_word(i)
            # Remove the word
            #lexicon.remove_one(centre)
            restaurant.remove_customer(centre) #
            # Add the words
            utterance.line_boundaries[i] = True
            #lexicon.add_one(left)
            #lexicon.add_one(right)
            restaurant.add_customer(left) #
            restaurant.add_customer(right) #

    def put_non_boundary(self, i, state, utterance):
        '''Put a non-boundary (0) at position i.

        This implies to remove and then add the adequate words.
        '''
        if not utterance.line_boundaries[i]: # No boundary at the i-th position ('yes' case)
            pass
            #lexicon.remove_one(left)
            #lexicon.remove_one(right)
        else: # No boundary at the i-th position ('no' case)
            restaurant = state.restaurant #
            left = utterance.left_word(i)
            right = utterance.right_word(i)
            centre = utterance.centre_word(i)
            # Remove the word
            #restaurant.remove_customer(centre) #
            restaurant.remove_customer(left) #
            restaurant.remove_customer(right) #
            # Add the words
            utterance.line_boundaries[i] = False
            #restaurant.add_customer(left) #
            #restaurant.add_customer(right) #
            restaurant.add_customer(centre) #

    def sample(self, temp):
        #utils.check_equality(len(self.utterances), self.n_utterances)
        #for utterance in self.utterances: #
        for i_utt in range(self.n_utterances):
            utt_word = self.word_state.utterances[i_utt]
            utt_morph = self.morph_state.utterances[i_utt]
            #utils.check_equality(len(self.line_boundaries), len(self.sentence))
            utils.check_equality(len(utt_word.line_boundaries),
                                 len(utt_morph.line_boundaries))
            for i in range(len(utt_word.line_boundaries) - 1):
                if self.word_then_morph:
                    utt_word.sample_one(i, self.word_state, temp)
                    if utt_word.line_boundaries[i] == True: # i is a word boundary
                        self.put_boundary(i, self.morph_state, utt_morph)
                    else: # i is not a word boundary
                        utt_morph.sample_one(i, self.morph_state, temp)
                else: # Morpheme then words
                    utt_morph.sample_one(i, self.morph_state, temp)
                    if utt_morph.line_boundaries[i] == True: # if morpheme boundary
                        utt_word.sample_one(i, self.word_state, temp)
                    else: # i is not a morpheme boundary
                        # Remove word boundaries
                        self.put_non_boundary(i, self.word_state, utt_word)

        # Built-in hyperparameter sampling
        self.word_state.alpha_1, self.word_state.discount = \
                self.word_hyper_sample.sample_hyperparameter(self.word_state)
        self.morph_state.alpha_1, self.morph_state.discount = \
                self.morph_hyper_sample.sample_hyperparameter(self.morph_state)

    def get_segmented(self):
        '''Generate the segmented text with the current state of the boundaries.

        Two-level version, with morphemes separated by -.
        '''
        segmented_text_list = []
        #utils.check_equality(len(self.utterances), len(self.unsegmented_list))
        utils.check_equality(len(self.morph_state.utterances), self.n_utterances)
        for i in range(self.n_utterances):
            segmented_line_list = []
            unsegmented_line = self.morph_state.unsegmented_list[i]
            word_boundaries_line = self.word_state.utterances[i].line_boundaries
            morph_boundaries_line = self.morph_state.utterances[i].line_boundaries
            beg = 0
            pos = 0
            line_length = len(unsegmented_line)
            utils.check_equality(len(word_boundaries_line), line_length)
            utils.check_equality(len(morph_boundaries_line), line_length)
            #utils.check_equality(len(self.boundaries[i]), len(unsegmented_line))
            #for boundary in boundaries_line:
            word = []
            for j in range(line_length):
                word_boundary = word_boundaries_line[j]
                morph_boundary = morph_boundaries_line[j]
                if morph_boundary and word_boundary: # Word boundary
                    word.append(unsegmented_line[beg:(pos + 1)])
                    beg = pos + 1
                    segmented_line_list.append('-'.join(word))
                    word = [] # Start new word
                    #beg = pos + 1
                #pos += 1
                elif morph_boundary: # Morpheme boundary
                    word.append(unsegmented_line[beg:(pos + 1)])
                    beg = pos + 1
                else:
                    pass
                pos += 1
            # Convert list of words into a string sentence
            segmented_line = ' '.join(segmented_line_list)
            segmented_text_list.append(segmented_line)
        return '\n'.join(segmented_text_list) #segmented_text


class WordLevelRestaurant(Restaurant):
    #def __init__(self, alpha_1, discount, state, seed=42):

    def phi(self, hier_word):
        '''numer_base function from HierarchicalUtterance.'''
        #base = self.customers[word] - (self.discount * self.tables[word])
        #base += ((self.discount * self.n_tables) + self.alpha_1) \
        #        * self.state.p_word(word)
        #return base
        word = hier_word.sentence
        # Position is used to find the boundary section
        #if word not in state.restaurant.customers: # If the word is not in the lexicon
        #    base = 0
        #else: # The word is in the lexicon/restaurant
        base = self.customers[word] - (self.discount * self.tables[word])
        base += ((self.discount * self.n_tables) + self.alpha_1) \
                * self.state.p_0(hier_word) #, self.state) # here, not p_word()
        return base

    def add_customer(self, hier_word):
        '''Assign a customer (word) to a table in the restaurant.

        Hier, it is not a string (word) but the HierarchicalWord object
        (hier_word).
        '''
        utils.check_equality(len(self.customers.keys()), len(self.restaurant.keys()))
        word = hier_word.sentence
        if word in self.restaurant.keys(): # Add the customer to a table (possibly new)
            n_customers_w = self.customers[word]
            n_tables_w = self.tables[word]
            random_value = self.random_gen.random()
            new_customer = random_value * self.phi(hier_word) #self.phi(word)
            #new_customer = random_value * (n_customers_w + self.alpha_1)
            utils.check_equality(self.tables[word], len(self.restaurant[word]))
            if (new_customer > (n_customers_w - (self.discount * n_tables_w))):
                # Open a new table
                self.open_table(word, False)
            else: # Add the new customer in an existing table
                cumulative_sum = 0
                for k in range(n_tables_w):
                    cumulative_sum += (self.restaurant[word][k] - self.discount)
                    if new_customer <= cumulative_sum: # Add the customer to that table
                        self.restaurant[word][k] += 1
                        break
                    else:
                        pass
            self.customers[word] += 1
            #utils.check_equality(self.customers[word], n_customers + 1)

        else: # Open a new table for a new word
            self.customers[word] = 1
            self.open_table(word, True)

        self.n_customers += 1

    #def remove_customer(self, word):

    #def open_table(self, word, new_word=False):

    #def close_table(self, word, k, last_table=False):

    def get_morpheme_boundaries(self, sentence_word, sentence_morph):
        '''Get morpheme boundaries from a list of words and one of morphemes.'''
        boundaries = []
        morph_index = 0
        for word in sentence_word:
            word_in_morpheme = []
            while len(''.join(word_in_morpheme)) < len(word):
                word_in_morpheme.append(sentence_morph[morph_index])
                morph_index += 1
            #print(''.join(word_stock) == word)
            utils.check_equality(''.join(word_in_morpheme), word)
            morph_boundaries = []
            for morph in word_in_morpheme:
                #morph_boundaries = [False] * (len(morph) - 1) + [True]
                morph_boundaries.extend([False] * (len(morph) - 1) + [True])
            boundaries.append(morph_boundaries)
        utils.check_equality(len(boundaries), len(sentence_word))
        return boundaries

    def init_tables(self, text_word, text_morph):
        '''Initialise the tables in the restaurant with the given text.

        Here, HierarchicalWord objects are needed.'''
        length_text = len(text_word)
        utils.check_equality(length_text, len(text_morph))
        for i in range(length_text):
        #for line in text:
            #line_word = text_word[i]
            #line_morph = text_morph[i]
            #split_line = utils.line_to_word(line)
            #split_line_word = utils.line_to_word(line_word)
            #split_line_morph = utils.line_to_word(line_morph)
            line_word = utils.line_to_word(text_word[i])
            line_morph = utils.line_to_word(text_morph[i])
            morph_boundaries = self.get_morpheme_boundaries(line_word, line_morph)
            length_line = len(line_word)
            for j in range(length_line):
                hier_word = HierarchicalWord(line_word[j], morph_boundaries[j])
                self.add_customer(hier_word)


class HierarchicalTwoLevelState(PYPState): # Information on the whole document
    def __init__(self, data, discount, alpha_1, p_boundary,
                 discount_m, alpha_m, seed=42):
        # State parameters - word level
        self.discount = discount
        utils.check_value_between(discount, 0, 1)
        self.alpha_1 = alpha_1
        self.p_boundary = p_boundary
        utils.check_probability(p_boundary)
        # State parameters - morpheme level
        self.discount_m = discount_m
        utils.check_value_between(discount_m, 0, 1)
        self.alpha_m = alpha_m

        self.beta = 2 # Hyperparameter?

        logging.info(f'Word level:\t discount: {self.discount:.1f}, alpha_1: '
                     f'{self.alpha_1:d}, p_boundary: {self.p_boundary:.1f}')
        logging.info(f'Morpheme level:\t discount: {self.discount_m:.1f},'
                     f' alpha_1: {self.alpha_m:d}')

        self.seed = seed

        # data is a text segmented in morphemes with '-'
        self.data_word = utils.morpheme_gold_segment(data, False) # Word level
        self.data_morph = utils.morpheme_gold_segment(data, True) # Morpheme level

        # Data and Utterance object
        self.unsegmented = utils.unsegmented(self.data_word) #data
        self.unsegmented_list = utils.text_to_line(self.unsegmented)

        # Variable to store alphabet, utterance, and lexicon information
        self.utterances = [] # Stored Utterance objects

        for unseg_line in self.unsegmented_list: # rewrite with correct variable names
            # do next_reference function
            utterance = HierarchicalUtterance(unseg_line, self.p_boundary, seed)
            self.utterances.append(utterance)

        self.n_utterances = len(self.utterances) # Number of utterances

        # Lexicon object (Counter)
        #self.word_counts = Lexicon() # Word counter
        init_segmented_list = utils.text_to_line(self.get_segmented())

        # Alphabet (list of letters)
        self.alphabet = utils.delete_value_from_vector(list(set(self.unsegmented)), '\n')
        self.alphabet_size = len(self.alphabet)

        # Phoneme probability (dictionary)
        self.phoneme_ps = dict()
        #self.init_probs() # How to initialise the boundaries (here: random)
        self.init_phoneme_probs()

        # Morpheme restaurant
        init_segmented_m_list = utils.text_to_line(self.get_segmented_morph())
        self.restaurant_m = Restaurant(self.alpha_m, self.discount_m, self, self.seed)
        self.restaurant_m.init_tables(init_segmented_m_list)
        # Restaurant object to count the number of tables (dict)
        self.restaurant = Restaurant(self.alpha_1, self.discount, self, self.seed)
        self.restaurant = WordLevelRestaurant(self.alpha_1, self.discount, self, self.seed)
        self.restaurant.init_tables(init_segmented_list, init_segmented_m_list)
        logging.debug(f'{self.restaurant.n_tables} tables initially (word)')
        logging.debug(f'{self.restaurant_m.n_tables} tables initially (morpheme)')

        self.character_model = dict() # For P(morpheme) (string probability)


    #def init_phoneme_probs(self):

    # Probabilities
    #def p_cont(self):

    #def p_word(self, string):
    #    '''p_word from PYPState with a memory to store already-seen strings.'''
    #    if string in self.character_model:
    #        return self.character_model[string]
    #    else:
    #        #p = 1
    #        p = ((1 - self.p_boundary) ** (len(string) - 1)) * self.p_boundary
    #        for letter in string:
    #            p = p * self.phoneme_ps[letter]
    #        #p = p * ((1 - self.p_boundary) ** (len(string) - 1)) * self.p_boundary
    #        self.character_model[string] = p
    #        return p

    def p_0(self, hier_word):
        # This function has to be after sample_morphemes_in_words.
        p = 1
        morpheme_list = hier_word.decompose()
        ## sample the word -> access to the decomposition of the word
        for morpheme in morpheme_list:
            p = p * hier_word.p_morph(morpheme, self) #state)
        return p

    # Sampling
    #def sample(self, temp):

    #def get_segmented(self):

    def get_segmented_morph(self):
        '''Generate the segmented text with the current state of the boundaries.

        Morpheme level ###.
        Can be simplified
        '''
        segmented_text_list = []
        utils.check_equality(len(self.utterances), len(self.unsegmented_list))
        for i in range(len(self.utterances)):
            segmented_line_list = []
            unsegmented_line = self.unsegmented_list[i]
            boundaries_line = self.utterances[i].morph_boundaries
            beg = 0
            pos = 0
            utils.check_equality(len(boundaries_line), len(unsegmented_line))
            #utils.check_equality(len(self.boundaries[i]), len(unsegmented_line))
            for boundary in boundaries_line:
                if boundary: # If there is a boundary
                    segmented_line_list += [unsegmented_line[beg:(pos + 1)]]
                    beg = pos + 1
                pos += 1
            # Convert list of words into a string sentence
            segmented_line = ' '.join(segmented_line_list)
            segmented_text_list.append(segmented_line)
        return '\n'.join(segmented_text_list) #segmented_text

    def get_two_level_segmentation(self):
        '''Generate the segmented text with the current state of the boundaries.

        Two-level version, with morphemes separated by -.
        get_segmented() function from the TwoLevelState object.
        '''
        segmented_text_list = []
        #tils.check_equality(len(self.morph_state.utterances), self.n_utterances)
        for i in range(self.n_utterances):
            segmented_line_list = []
            unsegmented_line = self.unsegmented_list[i]
            word_boundaries_line = self.utterances[i].line_boundaries
            morph_boundaries_line = self.utterances[i].morph_boundaries
            beg = 0
            pos = 0
            line_length = len(unsegmented_line)
            utils.check_equality(len(word_boundaries_line), line_length)
            utils.check_equality(len(morph_boundaries_line), line_length)
            #utils.check_equality(len(self.boundaries[i]), len(unsegmented_line))
            #for boundary in boundaries_line:
            word = []
            for j in range(line_length):
                word_boundary = word_boundaries_line[j]
                morph_boundary = morph_boundaries_line[j]
                if morph_boundary and word_boundary: # Word boundary
                    word.append(unsegmented_line[beg:(pos + 1)])
                    beg = pos + 1
                    segmented_line_list.append('-'.join(word))
                    word = [] # Start new word
                elif morph_boundary: # Morpheme boundary
                    word.append(unsegmented_line[beg:(pos + 1)])
                    beg = pos + 1
                else:
                    pass
                pos += 1
            # Convert list of words into a string sentence
            segmented_line = ' '.join(segmented_line_list)
            segmented_text_list.append(segmented_line)
        return '\n'.join(segmented_text_list) #segmented_text


# Utterance in unigram case
class HierarchicalUtterance(PYPUtterance): # Information on one utterance of the document
    def __init__(self, sentence, p_segment, seed=42):
        self.sentence = sentence # Unsegmented utterance # Char
        self.p_segment = p_segment
        utils.check_probability(p_segment)

        self.line_boundaries = [] # Word-level boundaries
        self.morph_boundaries = [] # Morpheme-level boundaries
        self.init_boundary()
        self.random_gen = random.Random(seed)
        self.init_morph_boundary()


    #def init_boundary(self): # Random case only

    def init_morph_boundary(self): # Random case only
        for i in range(len(self.sentence) - 1):
            if self.line_boundaries[i]:
                self.morph_boundaries.append(True)
            else:
                rand_val = self.random_gen.random()
                #rand_val = random.random()
                if rand_val < self.p_segment:
                    self.morph_boundaries.append(True)
                else:
                    self.morph_boundaries.append(False)
        self.morph_boundaries.append(True)

    def numer_base(self, hier_word, state):
        word = hier_word.sentence
        # Position is used to find the boundary section
        if word not in state.restaurant.customers: # If the word is not in the lexicon
            base = 0
        else: # The word is in the lexicon/restaurant
            base = state.restaurant.customers[word] \
                   - (state.discount * state.restaurant.tables[word])
        base += ((state.discount * state.restaurant.n_tables) + state.alpha_1) \
                * state.p_0(hier_word) #self.p_0(hier_word, state) # here, not p_word()
        #print('numer_base: ', base)
        return base

    #def numer_base_morph(self, morph, state):
    #    '''Compute the numerator of the probability of a morpheme'''
    #    if word not in state.restaurant_m.customers: # If the word is not in the lexicon
    #        base = 0
    #    else: # The word is in the lexicon/restaurant
    #        base = state.restaurant_m.customers[morph] \
    #               - (state.discount_m * state.restaurant_m.tables[morph])
    #    base += ((state.discount_m * state.restaurant_m.n_tables) + state.alpha_m) \
    #            * state.p_word(morph) # uniform unigram character model
        #print('numer_base: ', base)
    #    return base

    #def p_morph(self, morpheme, state):
    #    '''Compute the probability of a morpheme'''
    #    denom = state.restaurant_m.n_customers + state.alpha_m
    #    return self.numer_base_morph(morpheme, state) / denom

    #def p_0(self, hier_word, state):
    #    # This function has to be after sample_morphemes_in_words.
    #    p = 1
    #    morpheme_list = hier_word.decompose()
    #    ## sample the word -> access to the decomposition of the word
    #    for morpheme in morpheme_list:
    #        p = p * hier_word.p_morph(morpheme, state)
    #    return p

    #def left_word(self, i):

    #def right_word(self, i):

    #def centre_word(self, i):

    def find_left_right_centre_words(self, i):
        '''Find the left, right, and centre words with morpheme boundaries.'''
        self.prev = self.prev_boundary(i)
        self.next = self.next_boundary(i)
        #left = self.left_word(i)
        #right = self.right_word(i)
        #centre = self.centre_word(i)
        utils.check_value_between(i, 0, len(self.sentence) - 1) # No last pos
        left = self.sentence[(self.prev + 1):(i + 1)]
        right = self.sentence[(i + 1):(self.next + 1)]
        centre = self.sentence[(self.prev + 1):(self.next + 1)]
        # Morpheme boundaries
        left_m_boundaries = self.morph_boundaries[(self.prev + 1):i] # Check values
        #(not i + 1 beacuse of final True)
        left_m_boundaries.append(True)
        right_m_boundaries = self.morph_boundaries[(i + 1):(self.next + 1)] # Check values
        centre_m_boundaries = self.morph_boundaries[(self.prev + 1):(self.next + 1)] # Check values
        self.left_word = HierarchicalWord(left, left_m_boundaries)
        self.right_word = HierarchicalWord(right, right_m_boundaries)
        self.centre_word = HierarchicalWord(centre, centre_m_boundaries)

    def sample_morphemes_in_words(self, state, temp):
        '''Sampling the left, right, and centre words into morphemes.'''
        # This function needs to be after find_left_right_centre_words.
        # Sampling
        self.left_word.sample_morph(state, temp)
        self.right_word.sample_morph(state, temp)
        self.centre_word.sample_morph(state, temp)

    def update_morph_boundaries(self, new_boundaries):
        self.morph_boundaries[(self.prev + 1):(self.next + 1)] = new_boundaries

    #def sample(self, state, temp):

    def sample_one(self, i, state, temp):
        restaurant = state.restaurant #
        #left = self.left_word(i)
        #right = self.right_word(i)
        #centre = self.centre_word(i)
        self.find_left_right_centre_words(i)
        left = self.left_word.sentence
        right = self.right_word.sentence
        centre = self.centre_word.sentence
        ### boundaries is the boundary for the utterance only here
        if self.line_boundaries[i]: # Boundary at the i-th position ('yes' case)
            #print('yes case')
            restaurant.remove_customer(left)
            restaurant.remove_customer(right)
            # Remove the morphemes
            self.left_word.remove_morphemes(state.restaurant_m)
            self.right_word.remove_morphemes(state.restaurant_m)
        else: # No boundary at the i-th position ('no' case)
            #print('no case')
            restaurant.remove_customer(centre)
            # Remove the morphemes
            self.centre_word.remove_morphemes(state.restaurant_m)
        # Sample morphemes
        self.sample_morphemes_in_words(state, temp)

        denom = restaurant.n_customers + state.alpha_1
        #print('denom: ', denom)
        yes = state.p_cont() * self.numer_base(self.left_word, state) \
        * (self.numer_base(self.right_word, state) + utils.kdelta(left, right))
        yes = yes / (denom + 1)
        #print('yes: ', yes)
        no = self.numer_base(self.centre_word, state)
        #print('no: ', no)

        # Normalisation
        yes = yes / (yes + no)
        no = 1 - yes

        #if (yes < 0) or (no < 0): # Invalid value test
            #print(f'yes is negative: {yes}')
            #print(f' right: {right}, left: {left}, centre: {centre}')
            #print(f'or no is negative: {no}')
        # Annealing
        yes = yes ** temp
        #print('yes temp: ', yes)
        no = no ** temp
        p_yes = yes / (yes + no)
        random_value = random.random()
        if (random_value < p_yes):
            #print('Boundary case')
            new_boundaries = self.left_word.line_boundaries + self.right_word.line_boundaries
            # Update morpheme boundaries
            self.update_morph_boundaries(new_boundaries)
            # Add the selected morphemes in the morpheme restaurant
            self.left_word.add_morphemes(state.restaurant_m)
            self.right_word.add_morphemes(state.restaurant_m)
            # Word-level update
            self.line_boundaries[i] = True
            #restaurant.add_customer(left) #
            #restaurant.add_customer(right) #
            restaurant.add_customer(self.left_word) #
            restaurant.add_customer(self.right_word) #
        else:
            #print('No boundary case')
            # Update morpheme boundaries
            self.update_morph_boundaries(self.centre_word.line_boundaries)
            # Add the selected morphemes in the morpheme restaurant
            self.centre_word.add_morphemes(state.restaurant_m)
            # Word-level update
            self.line_boundaries[i] = False
            #restaurant.add_customer(centre) #
            restaurant.add_customer(self.centre_word) #

    #def prev_boundary(self, i):

    #def next_boundary(self, i):

# Word in unigram case
class HierarchicalWord(PYPUtterance): # Information on one utterance of the document
    def __init__(self, sentence, boundaries): #p_segment):
        self.sentence = sentence # Unsegmented utterance # Char
        #self.p_segment = p_segment
        #utils.check_probability(p_segment)
        #self.line_boundaries = [] # Here, morpheme-level boundaries
        self.line_boundaries = boundaries # Here, morpheme-level boundaries
        #self.init_boundary()


    #def init_boundary(self): # Random case only

    #def numer_base(self, word, state):

    def numer_base_morph(self, morph, state):
        '''Compute the numerator of the probability of a morpheme'''
        if morph not in state.restaurant_m.customers: # If the word is not in the lexicon
            base = 0
        else: # The word is in the restaurant
            base = state.restaurant_m.customers[morph] \
                   - (state.discount_m * state.restaurant_m.tables[morph])
        base += ((state.discount_m * state.restaurant_m.n_tables) + state.alpha_m) \
                * state.p_word(morph)
        #print('numer_base: ', base)
        return base

    def p_morph(self, morpheme, state):
        '''Compute the probability of a morpheme'''
        denom = state.restaurant_m.n_customers + state.alpha_m
        return self.numer_base_morph(morpheme, state) / denom

    #def left_word(self, i):

    #def right_word(self, i):

    #def centre_word(self, i):

    #def sample(self, state, temp):

    #def sample_one(self, i, state, temp):

    def sample_morph(self, state, temp):
        utils.check_equality(len(self.line_boundaries), len(self.sentence))
        for i in range(len(self.line_boundaries) - 1):
            self.sample_one_morph(i, state, temp)
            #self.sample_one(i, state, temp)

    def sample_one_morph(self, i, state, temp):
        restaurant = state.restaurant_m # Morpheme level restaurant
        left = self.left_word(i)
        right = self.right_word(i)
        centre = self.centre_word(i)
        ### boundaries is the boundary for the utterance only here
        # No morpheme is removed because of the remove_morphemes function.
        #if self.line_boundaries[i]: # Boundary at the i-th position ('yes' case)
            #print('yes case')
        #    restaurant.remove_customer(left)
        #    restaurant.remove_customer(right)
        #else: # No boundary at the i-th position ('no' case)
            #print('no case')
        #    restaurant.remove_customer(centre)

        denom = restaurant.n_customers + state.alpha_m
        #print('denom: ', denom)
        yes = state.p_cont() * self.numer_base_morph(left, state) \
        * (self.numer_base_morph(right, state) + utils.kdelta(left, right)) / (denom + 1)
        #print('yes: ', yes)
        no = self.numer_base_morph(centre, state)
        #print('no: ', no)

        # Normalisation
        yes = yes / (yes + no)
        no = 1 - yes

        #if (yes < 0) or (no < 0): # Invalid value test
            #print(f'yes is negative: {yes}')
            #print(f' right: {right}, left: {left}, centre: {centre}')
            #print(f'or no is negative: {no}')
        # Annealing
        yes = yes ** temp
        #print('yes temp: ', yes)
        no = no ** temp
        p_yes = yes / (yes + no)
        random_value = random.random()
        if (random_value < p_yes):
            #print('Boundary case')
            self.line_boundaries[i] = True
            # No morpheme is added here, because of the add_morphemes function.
            #restaurant.add_customer(left)
            #restaurant.add_customer(right)
        else:
            #print('No boundary case')
            self.line_boundaries[i] = False
            #restaurant.add_customer(centre)

    #def prev_boundary(self, i):

    #def next_boundary(self, i):

    def decompose(self):
        '''Decompose the word into a list of morphemes.'''
        morpheme_list = []
        beg = 0
        #pos = 0
        utils.check_equality(len(self.line_boundaries), len(self.sentence))
        #for boundary in self.line_boundaries:
        #    if boundary: # If there is a boundary
        #        morpheme_list.append(self.sentence[beg:(pos + 1)])
        #        beg = pos + 1
        #    pos += 1
        end = len(self.sentence)
        while beg < end:
            pos = self.line_boundaries.index(1, beg)
            morpheme_list.append(self.sentence[beg:(pos + 1)])
            beg = pos + 1
        return morpheme_list

    def remove_morphemes(self, restaurant):
        morpheme_list = self.decompose()
        for morpheme in morpheme_list:
            restaurant.remove_customer(morpheme)

    def add_morphemes(self, restaurant):
        morpheme_list = self.decompose()
        for morpheme in morpheme_list:
            restaurant.add_customer(morpheme)
