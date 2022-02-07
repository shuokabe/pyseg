import collections
import logging
import random

# Two-level model (word and morpheme)

logging.basicConfig(level = logging.DEBUG, format = '[%(asctime)s] %(message)s',
                    datefmt = '%d/%m/%Y %H:%M:%S')

#from pyseg.dpseg import State
from pyseg.pypseg import Restaurant, PYPState, PYPUtterance
from pyseg.supervised_dpseg import SupervisionHelper, SupervisedState
from pyseg.supervised_pypseg import SupervisedPYPState
from pyseg.two_level import (WordLevelRestaurant, HierarchicalTwoLevelState,
    HierarchicalUtterance, HierarchicalWord)
from pyseg.hyperparameter import Hyperparameter_sampling
from pyseg import utils


class SupervisionHelper(SupervisionHelper):
    '''Class to handle supervision. Two-level supervision case.'''
    #def __init__(self, supervision_data=None, supervision_method='none',
    #             supervision_parameter=0, supervision_boundary='none',
    #             supervision_boundary_parameter=0, verbose=True):
    def __init__(self, supervision_helper):
        # Supervision variable
        self.data = supervision_helper.data # dictionary or text file
        self.word_data = {}
        self.morph_data = {}
        self.adapt_sup_dictionaries()
        self.method = supervision_helper.method
        self.parameter = supervision_helper.parameter
        self.boundary_method = supervision_helper.boundary_method
        self.boundary_parameter = supervision_helper.boundary_parameter

        self.verbose = supervision_helper.verbose

        # Two-level segmentation (morpheme)
        #if self.boundary_method == 'morpheme':
        #    self.boundary_parameter = 1.0


    def adapt_sup_dictionaries(self):
        '''Create two levels of the supervision dictionary for each level.'''
        word_sup_dictionary_keys = []
        morph_sup_dictionary_keys = []
        for key, value in self.data.items():
            word_key = utils.morpheme_gold_segment(key, False)
            morpheme_key = utils.line_to_word(utils.morpheme_gold_segment(key, True))
            word_sup_dictionary_keys.extend([word_key] * value)
            morph_sup_dictionary_keys.extend(morpheme_key * value)
        self.word_data = collections.Counter(word_sup_dictionary_keys)
        self.morph_data = collections.Counter(morph_sup_dictionary_keys)

    #def supervision_logs(self):

    #def supervision_index(self, data_length=0):

    #def set_ngram_character_model(self, state_alphabet):

    def set_bigram_character_model(self, state_alphabet, level='word'):
        logging.info(f'For the {level} level:')
        logging.info('Phoneme distribution: dictionary supervision')
        logging.info(' Chosen initialisation method: bigram')
        # Create the bigram distirbution dictionary
        ngrams_in_dict_list = [] # List of ngrams in the supervision data
        #if level == 'word':
        #    supervision_dictionary = self.word_data
        #else: # level == 'morpheme'
        supervision_dictionary = self.morph_data
        #for word in self.data.keys():
        for word in supervision_dictionary.keys():
            considered_word = f'<{word:s}>'
            word_ngram_list = [considered_word[i:(i + 2)]
                               for i in range(len(considered_word) - 1)]
            ngrams_in_dict_list += word_ngram_list
        ngrams_in_dict = collections.Counter(ngrams_in_dict_list)

        # List of ngrams without the last letter
        letters_in_dict_list = [ngram[0] for ngram in ngrams_in_dict_list]
        letters_in_dict = collections.Counter(letters_in_dict_list)
        print('letters in dict', letters_in_dict)
        all_letters = state_alphabet + ['<', '>']
        #print(all_letters)
        list_all_ngram = [f'{first:s}{second:s}'
                          for first in all_letters for second in all_letters]

        # Smoothing
        epsilon = 0.01 # Smoothing parameter
        smooth_denominator = epsilon * (len(all_letters))
        #print('Smooth denominator:', smooth_denominator)
        phoneme_ps = dict()
        for ngram in list_all_ngram: #ngrams_in_dict.keys():
            phoneme_ps[ngram] = (ngrams_in_dict[ngram] + epsilon) \
                        / (letters_in_dict[ngram[0]] + smooth_denominator)
        #print('Ngram dictionary: {0}'.format(self.phoneme_ps))
        return phoneme_ps


class SupervisedHTLState(HierarchicalTwoLevelState): #PYPState):
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
        #self.sup = SupervisionHelper(supervision_helper)
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
                utterance = SupervisedHierUtterance(
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
                utterance = HierarchicalUtterance(unseg_line, self.p_boundary,
                                                  seed)
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

        # Mixture function
        if self.sup.method in ['mixture', 'mixture_bigram']:
            # Total number of words in the supervision dictionary
            self.n_words_sup = len(self.sup.word_data) #sum(self.sup.data.values())
            self.morph_n_words_sup = len(self.sup.morph_data) #sum(self.sup.data.values())
            if self.htl_level in ['both', 'word']: # Word level supervision
                print('Use the mixture function in p_0 (word).')
                self.p_0 = self.mixture_p_0
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


    def init_phoneme_probs(self):
        '''
        Computes (uniform distribution)
        ### TODO: complete the documentation
        '''
        # Skip part to calculate the true distribution of characters
        if self.sup.method in ['init_bigram', 'mixture_bigram']:
            # Supervision with a dictionary
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
        p = (1 - self.sup.parameter) * p
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
class SupervisedHierUtterance(HierarchicalUtterance): #PYPUtterance):
    '''Information on one utterance of the document'''
    def __init__(self, sentence, sup_sentence, morph_sup_sentence, p_segment,
                 random_gen, sup_boundary_method='none',
                 sup_boundary_parameter=0, supervision_bool=False,
                 htl_level='none', sup_data=dict(), seed=42):
        self.sentence = sentence # Unsegmented utterance # Char
        self.sup_sentence = sup_sentence # Supervision sentence (with spaces)
        # Morpheme-level supervision sentence (with spaces)
        self.morph_sup_sentence = morph_sup_sentence
        self.p_segment = p_segment
        utils.check_probability(p_segment)
        self.sentence_length = len(self.sentence)

        self.random_gen = random_gen

        self.line_boundaries = [] # Word-level boundaries
        self.morph_boundaries = [] # Morpheme-level boundaries
        self.init_boundary()
        self.morph_random_gen = random.Random(seed)
        self.init_morph_boundary()

        self.sup_boundary_method = sup_boundary_method
        self.sup_boundary_parameter = sup_boundary_parameter

        self.htl_level = htl_level
        if self.htl_level in ['both', 'morpheme']: # With morpheme supervision
            #print('Change find_lrc_words function for the supervised version')
            self.find_left_right_centre_words = self.supervised_find_lrc_words

        self.sup_boundaries = []
        self.morph_sup_boundaries = []
        if (self.sup_boundary_method == 'sentence') and not supervision_bool:
            self.sup_boundaries = [-1] * (self.sentence_length) #len(self.sentence))
            self.morph_sup_boundaries = [-1] * (self.sentence_length)
        #elif (self.sup_boundary_method == 'word'):
        #    self.sup_data = sup_data
        #    self.init_word_sup_boundaries()
        else:
            self.init_sup_boundaries()
        if self.htl_level in ['none', 'morpheme']: # No word boundary supervision
            self.sup_boundaries = [-1] * (self.sentence_length)
        if self.htl_level in ['none', 'word']: # No morpheme boundary supervision
            self.morph_sup_boundaries = [-1] * (self.sentence_length)
        utils.check_equality(self.sentence_length, len(self.sup_boundaries))


    #def init_boundary(self): # Random case only

    def init_sup_boundaries(self): # From SupervisedUtterance
        boundary_track = 0
        morph_boundary_track = 0
        unseg_length = self.sentence_length #len(self.sentence)
        for i in range(unseg_length - 1):
            if self.sup_boundary_method == 'random':
                rand_val = self.random_gen.random()
                if rand_val >= self.sup_boundary_parameter:
                    self.sup_boundaries.append(-1)
                    self.morph_sup_boundaries.append(-1)
                    if self.sup_sentence[boundary_track + 1] == ' ':
                        boundary_track += 1
                    if self.morph_sup_sentence[morph_boundary_track + 1] == ' ':
                        morph_boundary_track += 1
                    boundary_track += 1
                    morph_boundary_track += 1
                    continue
            if self.sup_sentence[boundary_track + 1] == ' ': # Boundary case
                if self.sup_boundary_method == 'true':
                    rand_val = self.random_gen.random()
                    if rand_val > self.sup_boundary_parameter:
                        self.sup_boundaries.append(-1)
                        self.morph_sup_boundaries.append(-1)
                        boundary_track += 2 #1
                        morph_boundary_track += 2
                        continue
                self.sup_boundaries.append(1)
                self.morph_sup_boundaries.append(1)
                boundary_track += 1
                morph_boundary_track += 1
            else: # No boundary case
                if self.sup_boundary_method in ['true', 'morpheme']:
                    self.sup_boundaries.append(-1)
                    if self.morph_sup_sentence[morph_boundary_track + 1] == ' ':
                        if self.sup_boundary_method == 'true':
                            rand_val = self.random_gen.random()
                            if rand_val > self.sup_boundary_parameter:
                                self.morph_sup_boundaries.append(-1)
                            else:
                                self.morph_sup_boundaries.append(1)
                            morph_boundary_track += 1
                    else:
                        self.morph_sup_boundaries.append(-1)
                else:
                    self.sup_boundaries.append(0)
                    if self.morph_sup_sentence[morph_boundary_track + 1] == ' ':
                        self.morph_sup_boundaries.append(1)
                        morph_boundary_track += 1
                    else:
                        self.morph_sup_boundaries.append(0)
            boundary_track += 1
            morph_boundary_track += 1
        self.sup_boundaries.append(1)
        self.morph_sup_boundaries.append(1)
        utils.check_equality(self.sentence_length, len(self.sup_boundaries))
        utils.check_equality(self.sentence_length, len(self.morph_sup_boundaries))

    def init_morph_boundary(self): # Random case only
        for i in range(self.sentence_length - 1):
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

    #def left_word(self, i):

    #def right_word(self, i):

    #def centre_word(self, i):

    #def find_left_right_centre_words(self, i):

    def supervised_find_lrc_words(self, i):
        '''Find the left, right, and centre words with morpheme boundaries.'''
        self.prev = self.prev_boundary(i)
        self.next = self.next_boundary(i)
        #left = self.left_word(i)
        #right = self.right_word(i)
        #centre = self.centre_word(i)
        ###utils.check_value_between(i, 0, self.sentence_length - 1) # No last pos###
        left = self.sentence[(self.prev + 1):(i + 1)]
        right = self.sentence[(i + 1):(self.next + 1)]
        centre = self.sentence[(self.prev + 1):(self.next + 1)]
        # Morpheme boundaries
        left_m_boundaries = self.morph_boundaries[(self.prev + 1):i] # Check values
        #(not i + 1 beacuse of final True)
        left_m_boundaries.append(True)
        # Check values!
        right_m_boundaries = self.morph_boundaries[(i + 1):(self.next + 1)]
        centre_m_boundaries = self.morph_boundaries[(self.prev + 1):(self.next + 1)]
        ### Use supervised versions
        # Morpheme boundaries
        prev_index = self.prev + 1
        next_index = self.next + 1
        left_m_sup_boundaries = self.morph_sup_boundaries[prev_index:i] # Check values
        #(not i + 1 beacuse of final True)
        #left_m_boundaries.append(True)
        # Check values!
        right_m_sup_boundaries = self.morph_sup_boundaries[(i + 1):next_index]
        centre_m_sup_boundaries = self.morph_sup_boundaries[prev_index:next_index]
        self.left_word = SupervisedHierWord(left, left_m_boundaries,
                                            left_m_sup_boundaries)
        self.right_word = SupervisedHierWord(right, right_m_boundaries,
                                             right_m_sup_boundaries)
        self.centre_word = SupervisedHierWord(centre, centre_m_boundaries,
                                              centre_m_sup_boundaries)

    def sample_morphemes_in_words(self, state, temp):
        '''Sampling the left, right, and centre words into morphemes.'''
        # This function needs to be after find_left_right_centre_words.
        # Sampling
        self.left_word.sample_morph(state, temp)
        self.right_word.sample_morph(state, temp)
        self.centre_word.sample_morph(state, temp)
        # Update morpheme list
        self.left_word.morpheme_list = self.left_word.decompose()
        self.right_word.morpheme_list = self.right_word.decompose()
        self.centre_word.morpheme_list = self.centre_word.decompose()

    #def update_morph_boundaries(self, new_boundaries):

    #def add_left_and_right(self, state):

    #def add_centre(self, state):

    #def sample(self, state, temp):

    def sample_one(self, i, state, temp):
        restaurant = state.restaurant #

        if self.line_boundaries[i] == self.sup_boundaries[i]:
            return # No sampling if correct boundary status
        else:
            pass
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

# Word in unigram case
class SupervisedHierWord(HierarchicalWord): #PYPUtterance):
    # Information on one utterance of the document
    def __init__(self, sentence, boundaries, sup_boundaries): #p_segment):
        self.sentence = sentence # Unsegmented utterance # Char
        #self.p_segment = p_segment
        #utils.check_probability(p_segment)
        self.sentence_length = len(self.sentence)
        #self.line_boundaries = [] # Here, morpheme-level boundaries
        self.line_boundaries = boundaries # Here, morpheme-level boundaries
        self.sup_boundaries = sup_boundaries # Supervision boundaries
        #self.init_boundary()
        self.morpheme_list = self.decompose()


    #def init_boundary(self): # Random case only

    #def numer_base(self, word, state):

    #def numer_base_morph(self, morph, state):

    #def p_morph(self, morpheme, state):

    #def left_word(self, i):

    #def right_word(self, i):

    #def centre_word(self, i):

    #def sample(self, state, temp):

    #def sample_one(self, i, state, temp):

    #def sample_morph(self, state, temp):

    def sample_one_morph(self, i, state, temp):
        if self.line_boundaries[i] == self.sup_boundaries[i]:
            return # No sampling if correct boundary status
        else:
            pass

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

        # Supervision
        if self.sup_boundaries[i] == 1:
            # No sampling if known boundary
            self.line_boundaries[i] = True
            #restaurant.add_customer(left)
            #restaurant.add_customer(right)
        elif self.sup_boundaries[i] == 0:
            # No sampling if known no boundary position
            self.line_boundaries[i] = False
            #restaurant.add_customer(centre)
        else: # self.sup_boundaries[i] == -1: # Sampling case
            denom = restaurant.n_customers + state.alpha_m
            #print('denom: ', denom)
            yes = state.p_cont_morph() * self.numer_base_morph(left, state) \
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

        ###if self.sup_boundaries[i] >= 0:###
        ###    utils.check_equality(self.sup_boundaries[i], self.line_boundaries[i])###

    #def prev_boundary(self, i):

    #def next_boundary(self, i):

    #def decompose(self):

    #def remove_morphemes(self, restaurant):

    #def add_morphemes(self, restaurant):
