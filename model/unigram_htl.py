import logging
import random

# Two-level model (word and morpheme)

logging.basicConfig(level = logging.DEBUG, format = '[%(asctime)s] %(message)s',
                    datefmt = '%d/%m/%Y %H:%M:%S')

#from pyseg.dpseg import State
from pyseg.model.pypseg import Restaurant #, PYPState, PYPUtterance
#from pyseg.supervised_dpseg import SupervisionHelper, SupervisedState
#from pyseg.supervised_pypseg import SupervisedPYPState
from pyseg.model.two_level import (HierarchicalTwoLevelState, HierarchicalUtterance,
WordLevelRestaurant)
from pyseg.model.supervised_two_level import (SupervisionHelper, SupervisedHTLState,
SupervisedHierUtterance)
#from pyseg.hyperparameter import Hyperparameter_sampling
from pyseg import utils


class UnigramHierarchicalTwoLevelState(HierarchicalTwoLevelState):
    # Information on the whole document
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
            utterance = UnigramHierarchicalUtterance(unseg_line, self.p_boundary, seed)
            self.utterances.append(utterance)

        self.n_utterances = len(self.utterances) # Number of utterances

        init_segmented_list = utils.text_to_line(self.get_segmented())

        # Alphabet (list of letters)
        self.alphabet = utils.delete_value_from_vector(list(set(self.unsegmented)), '\n')
        self.alphabet_size = len(self.alphabet)

        # Phoneme probability (dictionary)
        self.phoneme_ps = dict()
        #self.init_probs() # How to initialise the boundaries (here: random)
        self.init_phoneme_probs()
        self.word_length_ps = dict() # Test
        self.init_length_model() # Test

        # Morpheme restaurant
        init_segmented_m_list = utils.text_to_line(self.get_segmented_morph())
        self.character_model = dict() # For P(morpheme) (string probability)
        self.p_length = dict()
        self.restaurant_m = Restaurant(self.alpha_m, self.discount_m, self, self.seed)
        self.restaurant_m.init_tables(init_segmented_m_list)
        # Restaurant object to count the number of tables (dict)
        self.restaurant = WordLevelRestaurant(
                                self.alpha_1, self.discount, self, self.seed)
        self.restaurant.init_tables(init_segmented_list, init_segmented_m_list)
        logging.debug(f'{self.restaurant.n_tables} tables initially (word)')
        logging.debug(f'{self.restaurant_m.n_tables} tables initially (morpheme)')


    #def init_phoneme_probs(self):

    #def init_length_model(self):

    # Probabilities
    #def p_cont(self):

    #def p_cont_morph(self):

    #def p_word(self, string):

    def p_0(self, hier_word):
        # This function has to be after sample_morphemes_in_words.
        p = 1
        #morpheme_list = hier_word.decompose() #
        ## sample the word -> access to the decomposition of the word
        #for morpheme in morpheme_list:
        for morpheme in hier_word.morpheme_list:
            p = p * hier_word.p_morph(morpheme, self) #state)
        length = hier_word.sentence_length
        return p * self.word_length_ps[length]

    # Sampling
    #def sample(self, temp):

    #def get_segmented(self):

    #def get_segmented_morph(self):

    #def get_two_level_segmentation(self):


# Utterance in unigram case
class UnigramHierarchicalUtterance(HierarchicalUtterance):
    # Information on one utterance of the document
    #def __init__(self, sentence, p_segment, seed=42):


    #def init_boundary(self): # Random case only

    #def init_morph_boundary(self):

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

    #def left_word(self, i):

    #def right_word(self, i):

    #def centre_word(self, i):

    #def find_left_right_centre_words(self, i):

    #def sample_morphemes_in_words(self, state, temp):

    #def update_morph_boundaries(self, new_boundaries):

    #def add_left_and_right(self, state):

    #def add_centre(self, state):

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
        #yes = state.p_cont() * self.numer_base(self.left_word, state) \
        #* (self.numer_base(self.right_word, state) + utils.kdelta(left, right))
        #yes = yes / (denom + 1)
        # Unigram version for the word-level model
        yes = state.p_cont() * state.p_0(self.left_word) * state.p_0(self.right_word)
        #print('yes: ', yes)
        no = state.p_0(self.centre_word) #self.numer_base(self.centre_word, state)
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
        if (random_value < p_yes): # Boundary case
            self.line_boundaries[i] = True
            #restaurant.add_customer(self.left_word) #
            #restaurant.add_customer(self.right_word) #
            self.add_left_and_right(state)
        else: # No boundary case
            self.line_boundaries[i] = False
            #restaurant.add_customer(self.centre_word) #
            self.add_centre(state)

    #def prev_boundary(self, i):

    #def next_boundary(self, i):

# Supervised versions
class SupervisedUnigramHTLState(SupervisedHTLState): #PYPState):
    # Information on the whole document
    def __init__(self, data, discount, alpha_1, p_boundary, discount_m,
                 alpha_m, seed=42, supervision_helper=None, htl_level='none'):
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
        random_gen_sup = random.Random(self.seed)

        # Supervision helper
        self.sup = supervision_helper
        self.htl_level = htl_level
        if self.sup.method != 'none': # Dictionary supervision
            self.sup = SupervisionHelper(supervision_helper)
        self.sup.supervision_logs()

        # data is a text segmented in morphemes with '-'
        self.data_word = utils.morpheme_gold_segment(data, False) # Word level
        self.data_morph = utils.morpheme_gold_segment(data, True) # Morpheme level

        # Data and Utterance object
        self.unsegmented = utils.unsegmented(self.data_word) #data
        self.unsegmented_list = utils.text_to_line(self.unsegmented)

        # Variable to store alphabet, utterance, and lexicon information
        self.utterances = [] # Stored Utterance objects

        if self.sup.boundary_method != 'none': # Boundary supervision
            self.sup_boundaries = [] # Stored supervision boundaries
            self.morph_sup_boundaries = [] #
            sup_data_list = utils.text_to_line(self.data_word)
            morph_sup_data_list = utils.text_to_line(self.data_morph)
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
                morph_sup_line = morph_sup_data_list[i]
                utterance = SupervisedUnigramHierUtterance(
                    unseg_line, sup_line, morph_sup_line, self.p_boundary,
                    random_gen_sup, self.sup.boundary_method,
                    self.sup.boundary_parameter, supervision_bool,
                    self.htl_level)
                self.utterances.append(utterance)
                self.sup_boundaries.append(utterance.sup_boundaries)
                self.morph_sup_boundaries.append(utterance.morph_sup_boundaries)

            # Count number of supervision boundaries
            utils.count_supervision_boundaries(self.sup_boundaries)
            utils.count_supervision_boundaries(self.morph_sup_boundaries)
        else: # Dictionary supervision (or no supervision) case
            for unseg_line in self.unsegmented_list: # rewrite with correct variable names
                #utterance = HierarchicalUtterance(unseg_line, self.p_boundary,
                #                                  seed)
                utterance = UnigramHierarchicalUtterance(
                                unseg_line, self.p_boundary, seed)
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
        #self.init_probs() # How to initialise the boundaries (here: random)
        self.init_phoneme_probs()
        self.word_length_ps = dict() # Test
        self.init_length_model() # Test

        # Mixture function
        if self.sup.method in ['mixture', 'mixture_bigram']:
            # Total number of words in the supervision dictionary
            self.n_words_sup = len(self.sup.word_data) #sum(self.sup.data.values())
            self.morph_n_words_sup = len(self.sup.morph_data) #sum(self.sup.data.values())
            if self.htl_level in ['both', 'word']: # Word level supervision
                print('Use the mixture function in p_0 (word).')
                self.p_0 = self.mixture_p_0
                # Length model for words
                self.word_length_ps = dict()
                self.init_length_model()
                #self.init_length_model()
            if self.htl_level in ['both', 'morpheme']: # Morpheme level supervision
                print('Use the mixture function in p_word (morpheme).')
                self.p_word = self.mixture_p_word

        # Morpheme restaurant
        init_segmented_m_list = utils.text_to_line(self.get_segmented_morph())
        self.character_model = dict() # For P(morpheme) (string probability)
        self.p_length = dict()
        self.restaurant_m = Restaurant(self.alpha_m, self.discount_m, self, self.seed)
        self.restaurant_m.init_tables(init_segmented_m_list)
        # Restaurant object to count the number of tables (dict)
        #self.restaurant = Restaurant(self.alpha_1, self.discount, self, self.seed)
        self.restaurant = WordLevelRestaurant(self.alpha_1, self.discount, self, self.seed)
        self.restaurant.init_tables(init_segmented_list, init_segmented_m_list)
        logging.debug(f'{self.restaurant.n_tables} tables initially (word)')
        logging.debug(f'{self.restaurant_m.n_tables} tables initially (morpheme)')


    #def init_phoneme_probs(self):

    #def init_length_model(self, length_model='standard'):

    # Probabilities
    #def p_cont(self):

    #def p_cont_morph(self):

    def p_bigram_character_model(self, string):
        '''
        Probability from the bigram character model.
        It uses a dictionary as memory to store the values.
        '''
        #if string in self.character_model:
        #    return self.character_model[string]
        #else:
        p = 1
        n_ngram = 2 #
        considered_word = f'<{string:s}>'
        for i in range(len(considered_word) - n_ngram + 1):
            ngram = considered_word[i:(i + n_ngram)]
            p = p * self.phoneme_ps[ngram] #
        #self.character_model[string] = p
        return p

    def p_word(self, string):
        '''p_word with supervision (and memory from HTLState).'''
        ### Bigram character model
        if string in self.character_model:
            return self.character_model[string]
        elif self.sup.method in ['init_bigram', 'mixture_bigram']:
            p = self.p_bigram_character_model(string)
        else:
            #p = ((1 - self.p_boundary) ** (len(string) - 1)) * self.p_boundary
            # Length model
            length = len(string)
            if length in self.p_length:
                p = self.p_length[length]
            else:
                p = ((1 - self.p_boundary) ** (length - 1)) * self.p_boundary
                self.p_length[length] = p # Save
            # Character model
            for letter in string:
                p = p * self.phoneme_ps[letter]
            #p = p * ((1 - self.p_boundary) ** (len(string) - 1)) * self.p_boundary
        self.character_model[string] = p # Save
        return p

    def mixture_p_word(self, string):
        '''Mixture version of p_word above.'''
        ### Mixture function and bigram character model
        if string in self.character_model:
            return self.character_model[string]
        elif self.sup.method in ['init_bigram', 'mixture_bigram']:
            p = self.p_bigram_character_model(string)
        else:
            # Length model
            length = len(string)
            if length in self.p_length:
                p = self.p_length[length]
            else:
                p = ((1 - self.p_boundary) ** (length - 1)) * self.p_boundary
                self.p_length[length] = p # Save
            # Character model
            for letter in string:
                p = p * self.phoneme_ps[letter]
            #p = p * ((1 - self.p_boundary) ** (len(string) - 1)) * self.p_boundary
        # Exclusive part for mixture supervision
        #if self.sup.method in ['mixture', 'mixture_bigram']:
            #print('p before mixture:', p)
        p = (1 - self.sup.parameter) * p
        p += (self.sup.parameter / self.morph_n_words_sup) \
              * utils.indicator(string, self.sup.morph_data)
        self.character_model[string] = p # Save
        return p

    #def p_0(self, hier_word):

    def mixture_p_0(self, hier_word):
        '''p_0 (word) when the mixture function is used for supervision.'''
        # This function has to be after sample_morphemes_in_words.
        p = 1
        #morpheme_list = hier_word.decompose() #
        ## sample the word -> access to the decomposition of the word
        for morpheme in hier_word.morpheme_list:
            p = p * hier_word.p_morph(morpheme, self)
        # Exclusive part for mixture supervision
        #if self.sup.method in ['mixture', 'mixture_bigram']: # Mixture function
            #print('p before mixture:', p)
        # With a length model for words (mixture case)
        length = hier_word.sentence_length
        p = (1 - self.sup.parameter) * p * self.word_length_ps[length]
        #* self.word_length_ps.get(length, 10 ** (-6))
        # n_words_sup for words and sup_data for words
        p += (self.sup.parameter / self.n_words_sup) \
             * utils.indicator(hier_word.sentence, self.sup.word_data)
        return p

    # Sampling
    #def sample(self, temp):

    #def get_segmented(self):

    #def get_segmented_morph(self):

    #def get_two_level_segmentation(self):


# Utterance in unigram case
class SupervisedUnigramHierUtterance(SupervisedHierUtterance): #PYPUtterance):
    '''Information on one utterance of the document'''
    #def __init__(self, sentence, sup_sentence, morph_sup_sentence, p_segment,
    #             random_gen, sup_boundary_method='none',
    #             sup_boundary_parameter=0, supervision_bool=False,
    #             htl_level='none', sup_data=dict(), seed=42):


    #def init_boundary(self): # Random case only

    #def init_sup_boundaries(self):

    #def init_morph_boundary(self): # Random case only

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

    #def left_word(self, i):

    #def right_word(self, i):

    #def centre_word(self, i):

    #def find_left_right_centre_words(self, i):

    #def supervised_find_lrc_words(self, i):

    #def sample_morphemes_in_words(self, state, temp):

    #def update_morph_boundaries(self, new_boundaries):

    #def add_left_and_right(self, state):

    #def add_centre(self, state):

    #def sample(self, state, temp):

    def sample_one(self, i, state, temp):
        restaurant = state.restaurant #

        #if self.line_boundaries[i] == self.sup_boundaries[i]:
        #    return # No sampling if correct boundary status
        #else:
        #    pass
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

        if self.line_boundaries[i] == self.sup_boundaries[i]:
            if self.sup_boundaries[i] == 1:
                self.add_left_and_right(state)
            else: # self.sup_boundaries[i] == 0
                self.add_centre(state)
            return # No sampling if correct boundary status
        #else:
        #    pass

        # Supervision
        if self.sup_boundaries[i] == 1:
            # No sampling if known boundary
            #restaurant.add_customer(left) #
            #restaurant.add_customer(right) #
            self.line_boundaries[i] = True
            self.add_left_and_right(state)
        elif self.sup_boundaries[i] == 0:
            # No sampling if known no boundary position
            #restaurant.add_customer(centre) #
            self.line_boundaries[i] = False
            self.add_centre(state)
        else: # self.sup_boundaries[i] == -1: # Sampling case
            denom = restaurant.n_customers + state.alpha_1
            #print('denom: ', denom)
            #yes = state.p_cont() * self.numer_base(self.left_word, state) \
            #* (self.numer_base(self.right_word, state) + utils.kdelta(left, right))
            #yes = yes / (denom + 1)
            yes = state.p_cont() * state.p_0(self.left_word) \
                * state.p_0(self.right_word)
            #print('yes: ', yes)
            #no = self.numer_base(self.centre_word, state)
            no = state.p_0(self.centre_word)
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
                self.add_left_and_right(state)
            else:
                #print('No boundary case')
                self.line_boundaries[i] = False
                self.add_centre(state)

        ###if self.sup_boundaries[i] >= 0:###
        ###    utils.check_equality(self.sup_boundaries[i], self.line_boundaries[i])###

    #def prev_boundary(self, i):

    #def next_boundary(self, i):
