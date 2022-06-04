import logging
import random

from scipy.stats import norm

# Two-level model (word and morpheme)

logging.basicConfig(level = logging.DEBUG, format = '[%(asctime)s] %(message)s',
                    datefmt = '%d/%m/%Y %H:%M:%S')

#from pyseg.dpseg import State
#from pyseg.model.pypseg import Restaurant #, PYPState, PYPUtterance
from pyseg.model.two_level import (HierarchicalTwoLevelState, HierarchicalUtterance,
WordLevelRestaurant, HierarchicalWord)
from pyseg.model.supervised_two_level import (TwoLevelSupervisionHelper,
SupervisedHTLState, SupervisedHierUtterance)
from pyseg.model.alternative_htl import (AltWordLevelRestaurant,
MorphemeLevelRestaurant, SimpleSeenWordsAndSeg,
SimpleAltHierarchicalTwoLevelState, SimpleAltHierUtterance,
SimpleAltHierarchicalWord)
from pyseg import utils


# Simpler version of the alternative HTL model
class SupervisedSimpleAltHTLState(SimpleAltHierarchicalTwoLevelState):
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
            self.sup = TwoLevelSupervisionHelper(supervision_helper)
        self.sup.supervision_logs()

        # data is a text segmented in morphemes with '-'
        self.data_word = utils.morpheme_gold_segment(data, False) # Word level
        self.data_morph = utils.morpheme_gold_segment(data, True) # Morpheme level

        # Data and Utterance object
        self.unsegmented = utils.unsegmented(self.data_word) #data
        self.unsegmented_list = utils.text_to_line(self.unsegmented)

        # Variable to store alphabet, utterance, and lexicon information
        self.utterances = [] # Stored Utterance objects

        # # Exclusive to atlHTL
        self.n_words_for_morph = 0 # For p_cont_morph
        self.seen_word_seg = SimpleSeenWordsAndSeg()
        self.segmented_m_list = [] # Morphemes to add in the morpheme restaurant

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
                utterance = SupervisedSimpleAltHierUtterance(
                    unseg_line, sup_line, morph_sup_line, self.p_boundary,
                    random_gen_sup, self.sup.boundary_method,
                    self.sup.boundary_parameter, supervision_bool,
                    self.htl_level)
                self.init_morpheme_level_boundaries(utterance) #
                self.utterances.append(utterance)
                self.sup_boundaries.append(utterance.sup_boundaries)
                self.morph_sup_boundaries.append(utterance.morph_sup_boundaries)

            # Count number of supervision boundaries
            utils.count_supervision_boundaries(self.sup_boundaries)
            utils.count_supervision_boundaries(self.morph_sup_boundaries)

        else: # Dictionary supervision (or no supervision) case
            for unseg_line in self.unsegmented_list:
                utterance = SimpleAltHierUtterance(unseg_line, self.p_boundary, seed)
                # Morpheme-level boundary initialisation for altHTL
                # Instead of init_morph_boundary()
                self.init_morpheme_level_boundaries(utterance)
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
        self.word_length_ps = dict()
        self.morpheme_length_ps = dict() # Exclusive to the altHTL model
        self.init_length_models() # Test: length modelS (plural)

        # Mixture function
        if self.sup.method in ['mixture', 'mixture_bigram']:
            # Total number of words in the supervision dictionary
            if self.htl_level in ['both', 'word']: # Word level supervision
                print('Use the mixture function in p_0 (word).')
                self.p_0 = self.mixture_p_0
            if self.htl_level in ['both', 'morpheme']: # Morpheme level supervision
                print('Use the mixture function in p_word (morpheme).')
                self.p_word = self.mixture_p_word

        # Morpheme restaurant
        init_segmented_m_list = utils.text_to_line(self.get_segmented_morph())
        self.character_model = dict() # For P(morpheme) (string probability)
        #self.p_length = dict() # test
        self.restaurant_m = MorphemeLevelRestaurant(self.alpha_m,
                            self.discount_m, self, self.seed)
        self.restaurant_m.init_tables(self.segmented_m_list) #init_segmented_m_list)
        # Restaurant object to count the number of tables (dict)
        self.restaurant = AltWordLevelRestaurant(
                                self.alpha_1, self.discount, self, self.seed)
        self.restaurant.init_tables(init_segmented_list, init_segmented_m_list)
        logging.debug(f'{self.restaurant.n_tables} tables initially (word)')
        logging.debug(f'{self.restaurant_m.n_tables} tables initially (morpheme)')
        logging.debug(f'{self.restaurant.n_customers} tokens initially (word)')
        logging.debug(f'{self.restaurant_m.n_customers} tokens initially (morpheme)')

        # To count the number of morpheme updates
        self.count_morpheme_updates()

        #for word, segmentations in self.seen_words_and_seg.items():
        #    seen_segmentations = self.seen_words_and_seg[word]
        #    self.seen_words_and_seg[word] = list(set(seen_segmentations))
        #print(f'unique seen_words_and_seg {self.seen_words_and_seg}')
        #print(f'unique seen_words_and_seg {self.seen_word_seg.word_seg}')
        #utils.check_equality(set(self.seen_words_and_seg.keys()),
        #                    set(self.seen_word_seg.word_seg.keys()))
        #utils.check_equality(set(self.seen_words_and_seg.keys()),
        #                    set(self.restaurant.customers.keys()))


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

    #def init_length_model(self): # Test

    #def init_length_models(self): # Standard

    def init_morpheme_level_boundaries(self, utterance): #
        '''Initialise the morpheme level boundaires for uniHTL.

        This function is used after each initialisation of the
        SimpleAltHierUtterance object.
        Filling the self.seen_word_seg.word_seg object as follows:
            {'word': [segmentation as a list of bools]}
        '''
        word_list = utils.segment_sentence_with_boundaries(
                    utterance.sentence, utterance.line_boundaries)
        for word in word_list:
            #word_segmentation = utterance.random_morph_boundary_for_word(word)
            if (word not in self.seen_word_seg.word_seg): # New word
                word_segmentation = utterance.random_morph_boundary_for_word(word)
                morpheme_list = utils.segment_sentence_with_boundaries(
                            word, word_segmentation)
                # Add morphemes to segmented_m_list
                self.segmented_m_list.extend(morpheme_list)
                self.n_words_for_morph += 1
                # Add the word and segmentation to seen_words_and_seg
                #self.seen_words_and_seg[word] = [[word_segmentation, 1]]
                self.seen_word_seg.add_word_and_seg(word, word_segmentation)
            #elif (word_segmentation not in
            #    [pair[0] for pair in self.seen_words_and_seg[word]]):
            #elif (word_segmentation not in self.seen_word_seg.word_seg[word]):
                # Already seen word but new segmentation
                #morpheme_list = utils.segment_sentence_with_boundaries(
                #            word, word_segmentation)
                # Add morphemes to segmented_m_list
                #self.segmented_m_list.extend(morpheme_list)
                #self.n_words_for_morph += 1
                # Add the word and segmentation to seen_words_and_seg
                #self.seen_words_and_seg[word].append([word_segmentation, 1])
                #self.seen_word_seg.add_word_and_seg(word, word_segmentation)
            else: # Already seen word and segmentation
                # Do nothing and use an existing word segmentation
                word_segmentation = self.seen_word_seg.word_seg[word]
                #word_seg_list = self.seen_word_seg.word_seg[word]
                #total_word_freq = sum([seg_freq[1] for seg_freq in word_seg_list])
                #seg_freq_list = self.seen_word_seg.seg_freq[word]
                #total_word_freq = sum(self.seen_word_seg.seg_freq[word])
                #rand_val = utterance.morph_random_gen.random() * total_word_freq
                #cumul_freq = 0
                #for (segmentation, freq) in word_seg_list:
                #for i in range(len(word_seg_list)):
                #    (segmentation, freq) = word_seg_list[i], seg_freq_list[i]
                #    cumul_freq += freq
                #    if cumul_freq > rand_val:
                #        word_segmentation = segmentation
                #word_segmentation = word_seg_list[int(rand_val)]
                if word_segmentation == []:
                    print(f'No segmentation assigned for {word}')
                self.seen_word_seg.add_word_and_seg(word, word_segmentation)
                #self.seen_words_and_seg[word][i][1] += 1
            # Update the morpheme boundaries in utterance
            utterance.morph_boundaries.extend(word_segmentation)

    #def init_count_morpheme_updates(self):
    #def next_count_morpheme_updates(self):

    # Probabilities
    #def p_cont(self):

    def p_cont_morph(self): #
        n_words = self.restaurant_m.n_customers
        n_utt = self.seen_word_seg.n_word_and_seg # N_types?
        #self.n_words_for_morph #self.restaurant.n_customers
        p = (n_words - n_utt + 1 + self.beta / 2) / (n_words + 1 + self.beta)
        ###utils.check_probability(p)###
        return p

    def p_bigram_character_model(self, string):
        '''
        Probability from the bigram character model.
        It uses a dictionary as memory to store the values.
        '''
        p = 1
        #n_ngram = 2 #
        considered_word = f'<{string:s}>'
        for i in range(len(considered_word) - 1): #- n_ngram + 1):
            ngram = considered_word[i:(i + 2)]
            p = p * self.phoneme_ps[ngram] #
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
            p = self.morpheme_length_ps[length]
            # Character model
            for letter in string:
                p = p * self.phoneme_ps[letter]
        self.character_model[string] = p # Save
        return p

    def mixture_p_word(self, string):
        '''Mixture version of p_word above (morpheme-level).'''
        ### Mixture function and bigram character model
        if string in self.character_model:
            return self.character_model[string]
        elif self.sup.method in ['init_bigram', 'mixture_bigram']:
            p = self.p_bigram_character_model(string)
        else:
            # Length model
            length = len(string)
            p = self.morpheme_length_ps[length]
            # Character model
            for letter in string:
                p = p * self.phoneme_ps[letter]
        # Exclusive part for mixture supervision
        #if self.sup.method in ['mixture', 'mixture_bigram']:
            #print('p before mixture:', p)

        #p = (1 - self.sup.parameter) * p
        #p += (self.sup.parameter / self.morph_n_words_sup) \
        #      * utils.indicator(string, self.sup.morph_data)
        p = self.sup.mixture_morph(string, p)
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
        p = p * self.word_length_ps[length]
        #p = (1 - self.sup.parameter) * p * self.word_length_ps[length]
        ##* self.word_length_ps.get(length, 10 ** (-6))
        # n_words_sup for words and sup_data for words
        #p += (self.sup.parameter / self.n_words_sup) \
        #     * utils.indicator(hier_word.sentence, self.sup.word_data)
        p = self.sup.mixture_word(hier_word.sentence, p)
        return p

    # Sampling
    #def sample(self, temp):

    #def get_segmented(self):

    #def get_segmented_morph(self):

    #def get_two_level_segmentation(self):


# Utterance in unigram case
class SupervisedSimpleAltHierUtterance(SimpleAltHierUtterance):
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
        #self.init_morph_boundary() # Morpheme-level initialisation elsewhere

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
        pass

    #def random_morph_boundary_for_word(self, word):

    #def numer_base(self, hier_word, state):

    #def left_word(self, i):
    #def right_word(self, i):
    #def centre_word(self, i):

    def find_left_right_centre_words(self, i):
        '''Find the left, right, and centre words with morpheme boundaries.'''
        self.prev = self.prev_boundary(i)
        self.next = self.next_boundary(i)
        ###utils.check_value_between(i, 0, self.sentence_length - 1) # No last pos###
        left = self.sentence[(self.prev + 1):(i + 1)]
        right = self.sentence[(i + 1):(self.next + 1)]
        centre = self.sentence[(self.prev + 1):(self.next + 1)]
        # Morpheme boundaries
        left_m_boundaries = self.morph_boundaries[(self.prev + 1):i] # Check values
        #(not i + 1 beacuse of final True)
        left_m_boundaries.append(True)
        right_m_boundaries = self.morph_boundaries[(i + 1):(self.next + 1)] # Check values
        centre_m_boundaries = self.morph_boundaries[(self.prev + 1):(self.next + 1)] # Check
        self.left_word = SimpleAltHierarchicalWord(left, left_m_boundaries)
        self.right_word = SimpleAltHierarchicalWord(right, right_m_boundaries)
        self.centre_word = SimpleAltHierarchicalWord(centre, centre_m_boundaries)

    def supervised_find_lrc_words(self, i):
        '''Find the left, right, and centre words with morpheme boundaries.'''
        self.prev = self.prev_boundary(i)
        self.next = self.next_boundary(i)
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
        prev_index = self.prev + 1 # For easier notations
        next_index = self.next + 1
        left_m_sup_boundaries = self.morph_sup_boundaries[prev_index:i] # Check values
        #(not i + 1 beacuse of final True)
        #left_m_boundaries.append(True)
        # Check values!
        right_m_sup_boundaries = self.morph_sup_boundaries[(i + 1):next_index]
        centre_m_sup_boundaries = self.morph_sup_boundaries[prev_index:next_index]
        self.left_word = SupervisedSimpleAltHierarchicalWord(left,
                            left_m_boundaries, left_m_sup_boundaries)
        self.right_word = SupervisedSimpleAltHierarchicalWord(right,
                            right_m_boundaries, right_m_sup_boundaries)
        self.centre_word = SupervisedSimpleAltHierarchicalWord(centre,
                            centre_m_boundaries, centre_m_sup_boundaries)

    #def sample_morphemes_in_words(self, state, temp):

    #def update_morph_boundaries(self, new_boundaries):

    def add_left_and_right(self, state):
        '''Add the left and right words in the restaurants after sampling.'''
        #print('Boundary case')
        new_boundaries = self.left_word.line_boundaries + self.right_word.line_boundaries
        # Update morpheme boundaries
        self.update_morph_boundaries(new_boundaries)
        # Add segmentation of words and seg
        add_left = state.seen_word_seg.add_word_and_seg(
                    self.left_word.sentence, self.left_word.line_boundaries)
        add_right = state.seen_word_seg.add_word_and_seg(
                    self.right_word.sentence, self.right_word.line_boundaries)
        # Add the selected morphemes in the morpheme restaurant
        if add_left:
            state.morpheme_update[state.morpheme_update_index] += 1 # Count updates
            self.left_word.add_morphemes(state.restaurant_m)
        if add_right:
            state.morpheme_update[state.morpheme_update_index] += 1 # Count updates
            self.right_word.add_morphemes(state.restaurant_m)
        # Word-level update
        #self.line_boundaries[i] = True
        state.restaurant.add_customer(self.left_word) #
        state.restaurant.add_customer(self.right_word) #

    def add_centre(self, state):
        '''Add the centre word in the restaurants after sampling.'''
        #print('No boundary case')
        # Update morpheme boundaries
        self.update_morph_boundaries(self.centre_word.line_boundaries)
        # Add segmentation of words and seg
        add_c = state.seen_word_seg.add_word_and_seg(
                    self.centre_word.sentence, self.centre_word.line_boundaries)
        # Add the selected morphemes in the morpheme restaurant
        if add_c:
            state.morpheme_update[state.morpheme_update_index] += 1 # Count updates
            self.centre_word.add_morphemes(state.restaurant_m)
        # Word-level update
        #self.line_boundaries[i] = False
        state.restaurant.add_customer(self.centre_word) #

    #def sample(self, state, temp):

    def sample_one(self, i, state, temp):
        restaurant = state.restaurant #

        self.find_left_right_centre_words(i)
        left = self.left_word.sentence
        right = self.right_word.sentence
        centre = self.centre_word.sentence
        ### boundaries is the boundary for the utterance only here
        if self.line_boundaries[i]: # Boundary at the i-th position ('yes' case)
            #print('yes case')
            restaurant.remove_customer(left)
            restaurant.remove_customer(right)
            # Remove segmentation from seen_words_and_seg
            # Bools to indicate if morpheme should be removed too.
            remove_left = state.seen_word_seg.remove_word_and_seg(left,
                                self.left_word.line_boundaries)
            remove_right = state.seen_word_seg.remove_word_and_seg(right,
                                self.right_word.line_boundaries)
            # Remove the morphemes, when necessary
            if remove_left:
                self.left_word.remove_morphemes(state.restaurant_m)
            if remove_right:
                self.right_word.remove_morphemes(state.restaurant_m)
        else: # No boundary at the i-th position ('no' case)
            #print('no case')
            restaurant.remove_customer(centre)
            # Remove segmentation from seen_words_and_seg
            remove_centre = state.seen_word_seg.remove_word_and_seg(centre,
                                self.centre_word.line_boundaries)
            # Remove the morphemes
            if remove_centre:
                self.centre_word.remove_morphemes(state.restaurant_m)
        # Sample morphemes
        self.sample_morphemes_in_words(state, temp)

        if self.line_boundaries[i] == self.sup_boundaries[i]:
            if self.sup_boundaries[i] == 1:
                self.add_left_and_right(state)
            else: # self.sup_boundaries[i] == 0
                self.add_centre(state)
            return # No sampling if correct boundary status

        # Supervision
        if self.sup_boundaries[i] == 1:
            # No sampling if known boundary
            self.line_boundaries[i] = True
            self.add_left_and_right(state)
        elif self.sup_boundaries[i] == 0:
            # No sampling if known no boundary position
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
            if (random_value < p_yes): # Boundary case
                self.line_boundaries[i] = True
                self.add_left_and_right(state)
            else: # No boundary case
                self.line_boundaries[i] = False
                self.add_centre(state)

    #def prev_boundary(self, i):
    #def next_boundary(self, i):


# Word in unigram case
class SupervisedSimpleAltHierarchicalWord(SimpleAltHierarchicalWord):
    # Information on one utterance of the document
    def __init__(self, sentence, boundaries, sup_boundaries): #p_segment):
        self.sentence = sentence # Unsegmented utterance # Char
        self.sentence_length = len(self.sentence)
        self.line_boundaries = boundaries # Here, morpheme-level boundaries
        self.sup_boundaries = sup_boundaries # Supervision boundaries
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

    def sample_morph(self, state, temp):
        '''Specific to altHTL: if already seen word, (usually) no sampling.'''
        ###utils.check_equality(len(self.line_boundaries), self.sentence_length)###
        #len(self.sentence))
        seen = (self.sentence in state.seen_word_seg.word_seg)
        #state.seen_words_and_seg) # Initialisation
        if seen: # Already seen word
            #word_seg_list = state.seen_word_seg.word_seg[self.sentence]
            word_seg = state.seen_word_seg.word_seg[self.sentence]
            ##state.seen_words_and_seg[word]
            ##total_word_freq = sum([seg_freq[1] for seg_freq in word_seg_list])
            #total_word_freq = sum(state.seen_word_seg.seg_freq[self.sentence])
            word_freq = state.seen_word_seg.seg_freq[self.sentence]
            #probability = total_word_freq / (total_word_freq + state.alpha_1)
            #random_value = random.random() * total_word_freq # Seed?
            #if random_value > probability:
                # Sample from scratch
                #self.sample_from_scratch(state, temp)
            #else:
                # No segmentation
                #n_seg = len(word_seg_list)
                #cumul_freq = 0
                #for (segmentation, frequency) in word_seg_list:
                #for i in range(n_seg):
                    #cumul_freq += state.seen_word_seg.seg_freq[self.sentence][i]
                    #if random_value > cumul_freq:
                        #self.line_boundaries = word_seg_list[i]
            self.line_boundaries = word_seg
        else: # New word
            # Sample from scratch
            self.sample_from_scratch(state, temp)

    #def sample_from_scratch(self, state, temp):
        # Sample from scratch
        #self.line_boundaries = [0] * (self.sentence_length - 1) + [1]
        #for i in range(self.sentence_length - 1):
        #    self.sample_one_morph(i, state, temp)
            #self.sample_one(i, state, temp)

    def sample_one_morph(self, i, state, temp):
        if self.line_boundaries[i] == self.sup_boundaries[i]:
            return # No sampling if correct boundary status
        #else:
        #    pass

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
