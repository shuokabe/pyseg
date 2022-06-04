import logging
import random

from scipy.stats import norm

# Two-level model (word and morpheme)

logging.basicConfig(level = logging.DEBUG, format = '[%(asctime)s] %(message)s',
                    datefmt = '%d/%m/%Y %H:%M:%S')

#from pyseg.dpseg import State
from pyseg.model.pypseg import Restaurant #, PYPState, PYPUtterance
from pyseg.model.two_level import (HierarchicalTwoLevelState, HierarchicalUtterance,
WordLevelRestaurant, HierarchicalWord)
#from pyseg.model.supervised_two_level import (SupervisionHelper, SupervisedHTLState,
#SupervisedHierUtterance)
from pyseg import utils


class AltWordLevelRestaurant(WordLevelRestaurant):
    #def __init__(self, alpha_1, discount, state, seed=42):

    #def phi(self, hier_word):

    #def add_customer(self, hier_word):

    #def remove_customer(self, word):

    #def open_table(self, word, new_word=False):

    #def close_table(self, word, k, last_table=False):

    #def get_morpheme_boundaries(self, sentence_word, sentence_morph):

    def init_tables(self, text_word, text_morph):
        '''Initialise the tables in the restaurant with the given text.

        Here, HierarchicalWord objects are needed.'''
        length_text = len(text_word)
        utils.check_equality(length_text, len(text_morph))
        for i in range(length_text):
        #for line in text:
            line_word = utils.line_to_word(text_word[i])
            line_morph = utils.line_to_word(text_morph[i])
            morph_boundaries = self.get_morpheme_boundaries(line_word, line_morph)
            length_line = len(line_word)
            for j in range(length_line):
                #hier_word = HierarchicalWord(line_word[j], morph_boundaries[j])
                hier_word = AltHierarchicalWord(line_word[j], morph_boundaries[j])
                self.add_customer(hier_word)


class MorphemeLevelRestaurant(Restaurant):
    #def __init__(self, alpha_1, discount, state, seed=42):

    #def phi(self, word):

    #def add_customer(self, word):

    #def remove_customer(self, word):

    #def open_table(self, word, new_word=False):

    #def close_table(self, word, k, last_table=False):

    def init_tables(self, list):
        '''Initialise the tables in the restaurant with the given LIST.'''
        for morph in list:
            self.add_customer(morph)


class SimpleSeenWordsAndSeg:
    '''Track seen words. Only one segmentation per word.

    word_seg: {'word': [segmentation: list of bool]}
    seg_freq: {'word': int}
    Here, only the value in seg_freq corresponds to the frequency of the
    unique segmentation.
    '''
    def __init__(self):
        self.word_seg = dict()
        self.seg_freq = dict()

        self.n_word_and_seg = 0

    def add_word_and_seg(self, word, segmentation):
        '''Add one occurrence of a word and its segmentation.

        Morphemes are always added when the word is new.'''
        if word in self.word_seg: # Already seen word
            #word_seg_list = self.word_seg[word]
            #if segmentation in word_seg_list: # Already seen segmentation
            #index = word_seg_list.index(segmentation)
            self.seg_freq[word] += 1
            return False
            #else: # New segmentation
            #    word_seg_list.append(segmentation)
            #    self.seg_freq[word].append(1)
            #    self.n_word_and_seg += 1
            #    return True
        else: # New word
            self.word_seg[word] = segmentation
            self.seg_freq[word] = 1
            self.n_word_and_seg += 1
            return True

    def remove_word_and_seg(self, word, segmentation):
        '''Remove one occurrence of a word with a specific segmentation.

        Indicate if morphemes should be removed from the restaurant.'''
        # Necessarily a seen word and seen segmentation
        #word_seg_list = self.word_seg[word]
        #index = self.word_seg[word].index(segmentation)
        self.seg_freq[word] += -1
        if self.seg_freq[word] < 0:
            raise KeyError(f'The dictionary does not contain the word {word}.')
        elif self.seg_freq[word] == 0: # Remove the segmentation
            del self.word_seg[word]
            del self.seg_freq[word]
            #if self.seg_freq[word] == []: # Remove the word entirely
            #self.word_seg.pop(word)
            #self.seg_freq.pop(word)
            self.n_word_and_seg += -1
            return True
        else:
            return False


class SeenWordsAndSeg:
    '''Track seen words. Several segmentation per word possible.

    word_seg: {'word': [[segmentation: list of bool]]}
    seg_freq: {'word': [int]}
    Both dictionaries are related: the index in the list corresponds to
    the index in the other list.
    '''
    def __init__(self):
        self.word_seg = dict()
        self.seg_freq = dict()

        self.n_word_and_seg = 0

    def add_word_and_seg(self, word, segmentation):
        '''Add one occurrence of a word and its segmentation.

        Indicate if morphemes should be added to the restaurant.'''
        if word in self.word_seg: # Already seen word
            word_seg_list = self.word_seg[word]
            if segmentation in word_seg_list: # Already seen segmentation
                index = word_seg_list.index(segmentation)
                self.seg_freq[word][index] += 1
                return False
            else: # New segmentation
                word_seg_list.append(segmentation)
                self.seg_freq[word].append(1)
                self.n_word_and_seg += 1
                return True
        else: # New word
            self.word_seg[word] = [segmentation]
            self.seg_freq[word] = [1]
            self.n_word_and_seg += 1
            return True

    def remove_word_and_seg(self, word, segmentation):
        '''Remove one occurrence of a word with a specific segmentation.

        Indicate if morphemes should be removed from the restaurant.'''
        # Necessarily a seen word and seen segmentation
        #word_seg_list = self.word_seg[word]
        index = self.word_seg[word].index(segmentation)
        self.seg_freq[word][index] += -1
        if self.seg_freq[word][index] <= 0: # Remove the segmentation
            del self.word_seg[word][index]
            del self.seg_freq[word][index]
            if self.seg_freq[word] == []: # Remove the word entirely
                self.word_seg.pop(word)
                self.seg_freq.pop(word)
            self.n_word_and_seg += -1
            return True
        else:
            return False


class AlternativeHierarchicalTwoLevelState(HierarchicalTwoLevelState):
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

        # # Exclusive to atlHTL
        self.n_words_for_morph = 0 # For p_cont_morph
        #self.seen_words_and_seg = dict() # Seen words and their segmentation
        self.seen_word_seg = SeenWordsAndSeg()
        self.segmented_m_list = [] # Morphemes to add in the morpheme restaurant
        for unseg_line in self.unsegmented_list: # rewrite with correct variable names
            # do next_reference function
            utterance = AltHierUtterance(unseg_line, self.p_boundary, seed)
            # Morpheme-level boundary initialisation for altHTL
            # Instead of init_morph_boundary()
            self.init_morpheme_level_boundaries(utterance)
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
        self.word_length_ps = dict()
        self.morpheme_length_ps = dict() # Exclusive to the altHTL model
        self.init_length_models() # Test: length modelS (plural)

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
        self.init_count_morpheme_updates()
        #for word, segmentations in self.seen_words_and_seg.items():
        #    seen_segmentations = self.seen_words_and_seg[word]
        #    self.seen_words_and_seg[word] = list(set(seen_segmentations))
        #print(f'unique seen_words_and_seg {self.seen_words_and_seg}')
        #print(f'unique seen_words_and_seg {self.seen_word_seg.word_seg}')
        #utils.check_equality(set(self.seen_words_and_seg.keys()),
        #                    set(self.seen_word_seg.word_seg.keys()))
        #utils.check_equality(set(self.seen_words_and_seg.keys()),
        #                    set(self.restaurant.customers.keys()))

    #def init_phoneme_probs(self):

    #def init_length_model(self): # Test

    def init_length_models(self): # Test
        '''Length models for word-level and morpheme-level P_0.'''
        print(f'Standard length model for morphemes')
        max_length = max([len(sent) for sent in self.unsegmented_list])
        p_b = self.p_boundary
        self.morpheme_length_ps = {i: (((1 - p_b) ** (i - 1)) * p_b)
                               for i in range(max_length)}
        print(f'Morpheme length: {sum(self.morpheme_length_ps.values())}')

        print(f'Gaussian model for the number of morphemes in a word.')
        # For word-level P_0: number of morphemes in a word
        self.word_length_ps = {i: norm.pdf(i, loc = 2, scale = 1) # By default
                            for i in range(1, max_length)}
        print(f'Number of morphemes: {sum(self.word_length_ps.values())}')

    def init_morpheme_level_boundaries(self, utterance):
        '''Initialise the morpheme level boundaires for altHTL.

        This function is used after each initialisation of the AltHierUtterance
        object.
        Filling the self.seen_word_seg.word_seg object as follows:
            {'word': [segmentation as a list of bools]}
        '''
        word_list = utils.segment_sentence_with_boundaries(
                    utterance.sentence, utterance.line_boundaries)
        for word in word_list:
            word_segmentation = utterance.random_morph_boundary_for_word(word)
            if (word not in self.seen_word_seg.word_seg):# and \
                #(word not in self.seen_words_and_seg): # New word
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
            elif (word_segmentation not in self.seen_word_seg.word_seg[word]):
                # Already seen word but new segmentation
                morpheme_list = utils.segment_sentence_with_boundaries(
                            word, word_segmentation)
                # Add morphemes to segmented_m_list
                self.segmented_m_list.extend(morpheme_list)
                self.n_words_for_morph += 1
                # Add the word and segmentation to seen_words_and_seg
                #self.seen_words_and_seg[word].append([word_segmentation, 1])
                self.seen_word_seg.add_word_and_seg(word, word_segmentation)
            else: # Already seen word and segmentation
                # Do nothing and use an existing word segmentation
                word_segmentation = []
                word_seg_list = self.seen_word_seg.word_seg[word]
                #total_word_freq = sum([seg_freq[1] for seg_freq in word_seg_list])
                seg_freq_list = self.seen_word_seg.seg_freq[word]
                total_word_freq = sum(self.seen_word_seg.seg_freq[word])
                #rand_val = utterance.morph_random_gen.random() * total_word_freq
                rand_val = random.random() * total_word_freq # Test
                cumul_freq = 0
                #for (segmentation, freq) in word_seg_list:
                for i in range(len(word_seg_list)):
                    (segmentation, freq) = word_seg_list[i], seg_freq_list[i]
                    cumul_freq += freq
                    if cumul_freq > rand_val:
                        word_segmentation = segmentation
                #word_segmentation = word_seg_list[int(rand_val)]
                if word_segmentation == []:
                    print(f'No segmentation assigned for {word}')
                self.seen_word_seg.add_word_and_seg(word, word_segmentation)
                #self.seen_words_and_seg[word][i][1] += 1
            # Update the morpheme boundaries in utterance
            utterance.morph_boundaries.extend(word_segmentation)

    def init_count_morpheme_updates(self):
        '''Track the number of morpheme updates made by the model.'''
        self.morpheme_update = []
        self.morpheme_scratch = []
        self.morpheme_update_index = -1
        assert (set(self.seen_word_seg.word_seg.keys()) ==
        set(self.restaurant.restaurant.keys())), ('All the seen words are not '
        'in the restaurant.')

    def next_count_morpheme_updates(self):
        '''Increment the lists for the next batch of morpheme updates.'''
        self.morpheme_update.append(0)
        self.morpheme_scratch.append(0)
        self.morpheme_update_index += 1

    # Probabilities
    #def p_cont(self):

    def p_cont_morph(self):
        n_words = self.restaurant_m.n_customers
        n_utt = self.seen_word_seg.n_word_and_seg
        #self.n_words_for_morph #self.restaurant.n_customers
        p = (n_words - n_utt + 1 + self.beta / 2) / (n_words + 1 + self.beta)
        ###utils.check_probability(p)###
        return p

    def p_word(self, string):
        '''p_word from PYPState with a memory to store already-seen strings.

        For the probability of a morpheme.'''
        if string in self.character_model:
            return self.character_model[string]
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

    def p_0(self, hier_word):
        # This function has to be after sample_morphemes_in_words.
        p = 1
        #morpheme_list = hier_word.decompose() #
        ## sample the word -> access to the decomposition of the word
        for morpheme in hier_word.morpheme_list:
            p = p * hier_word.p_morph(morpheme, self) #state)
        # Number of morphemes in a word
        length = len(hier_word.morpheme_list) #hier_word.sentence_length
        return p * self.word_length_ps[length]

    # Sampling
    #def sample(self, temp):

    #def get_segmented(self):

    #def get_segmented_morph(self):

    #def get_two_level_segmentation(self):


# Utterance in unigram case
class AltHierUtterance(HierarchicalUtterance):
    # Information on one utterance of the document
    def __init__(self, sentence, p_segment, seed=42):
        self.sentence = sentence # Unsegmented utterance # Char
        self.p_segment = p_segment
        utils.check_probability(p_segment)
        self.sentence_length = len(self.sentence)

        self.line_boundaries = [] # Word-level boundaries
        self.morph_boundaries = [] # Morpheme-level boundaries
        self.init_boundary()
        #self.morph_random_gen = random.Random(seed)
        #self.init_morph_boundary() # Morpheme-level initialisation elsewhere


    #def init_boundary(self): # Random case only

    def init_morph_boundary(self): # Random case only
        pass

    def random_morph_boundary_for_word(self, word):
        # Random morpheme segmentation of a word for initialisation
        random_morph_boundaries = []
        for i in range(len(word) - 1):
            rand_val = random.random() #self.morph_random_gen.random() # Test
            if rand_val < self.p_segment:
                random_morph_boundaries.append(True)
            else:
                random_morph_boundaries.append(False)
        random_morph_boundaries.append(True)
        return random_morph_boundaries

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
        self.left_word = AltHierarchicalWord(left, left_m_boundaries)
        self.right_word = AltHierarchicalWord(right, right_m_boundaries)
        self.centre_word = AltHierarchicalWord(centre, centre_m_boundaries)

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
            #restaurant.add_customer(self.left_word) #
            #restaurant.add_customer(self.right_word) #
            self.add_left_and_right(state)
        else: # No boundary case
            self.line_boundaries[i] = False
            #restaurant.add_customer(self.centre_word) #
            self.add_centre(state)

    #def prev_boundary(self, i):

    #def next_boundary(self, i):


# Word in unigram case
class AltHierarchicalWord(HierarchicalWord):
    # Information on one utterance of the document
    #def __init__(self, sentence, boundaries):

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
            word_seg_list = state.seen_word_seg.word_seg[self.sentence]
            #state.seen_words_and_seg[word]
            #total_word_freq = sum([seg_freq[1] for seg_freq in word_seg_list])
            total_word_freq = sum(state.seen_word_seg.seg_freq[self.sentence])
            probability = total_word_freq / (total_word_freq + state.alpha_1)
            random_value = random.random() * total_word_freq # Seed?
            if random_value > probability:
                # Sample from scratch
                self.sample_from_scratch(state, temp)
            else:
                # No segmentation
                n_seg = len(word_seg_list)
                cumul_freq = 0
                #for (segmentation, frequency) in word_seg_list:
                for i in range(n_seg):
                    cumul_freq += state.seen_word_seg.seg_freq[self.sentence][i]
                    if random_value > cumul_freq:
                        self.line_boundaries = word_seg_list[i]
        else: # New word
            # Sample from scratch
            self.sample_from_scratch(state, temp)

    def sample_from_scratch(self, state, temp):
        # Sample from scratch
        state.morpheme_scratch[state.morpheme_update_index] += 1 # All sampled
        self.line_boundaries = [False] * (self.sentence_length - 1) + [True]
        for i in range(self.sentence_length - 1):
            self.sample_one_morph(i, state, temp)
            #self.sample_one(i, state, temp)

    #def sample_one_morph(self, i, state, temp):

    #def prev_boundary(self, i):
    #def next_boundary(self, i):

    #def decompose(self):

    #def remove_morphemes(self, restaurant):

    #def add_morphemes(self, restaurant):


# Simpler version of the alternative HTL model
class SimpleAltHierarchicalTwoLevelState(AlternativeHierarchicalTwoLevelState):
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

        # # Exclusive to atlHTL
        self.n_words_for_morph = 0 # For p_cont_morph
        self.seen_word_seg = SimpleSeenWordsAndSeg()
        self.segmented_m_list = [] # Morphemes to add in the morpheme restaurant
        for unseg_line in self.unsegmented_list: # rewrite with correct variable names
            # do next_reference function
            utterance = SimpleAltHierUtterance(unseg_line, self.p_boundary, seed)
            # Morpheme-level boundary initialisation for altHTL
            # Instead of init_morph_boundary()
            self.init_morpheme_level_boundaries(utterance)
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
        self.word_length_ps = dict()
        self.morpheme_length_ps = dict() # Exclusive to the altHTL model
        self.init_length_models() # Test: length modelS (plural)

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
        self.init_count_morpheme_updates()
        #self.morpheme_update, self.morpheme_scratch, self.morpheme_update_index

        #for word, segmentations in self.seen_words_and_seg.items():
        #    seen_segmentations = self.seen_words_and_seg[word]
        #    self.seen_words_and_seg[word] = list(set(seen_segmentations))
        #print(f'unique seen_words_and_seg {self.seen_words_and_seg}')
        #print(f'unique seen_words_and_seg {self.seen_word_seg.word_seg}')
        #utils.check_equality(set(self.seen_words_and_seg.keys()),
        #                    set(self.seen_word_seg.word_seg.keys()))
        #utils.check_equality(set(self.seen_words_and_seg.keys()),
        #                    set(self.restaurant.customers.keys()))

    #def init_phoneme_probs(self):

    #def init_length_model(self): # Test

    #def init_length_models(self): # Test #

    def init_morpheme_level_boundaries(self, utterance):
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

    #def p_cont_morph(self):
    #    n_words = self.restaurant_m.n_customers
    #    n_utt = self.seen_word_seg.n_word_and_seg # N_types?
    #    #self.n_words_for_morph #self.restaurant.n_customers
    #    p = (n_words - n_utt + 1 + self.beta / 2) / (n_words + 1 + self.beta)
        ###utils.check_probability(p)###
    #    return p

    #def p_word(self, string): #

    #def p_0(self, hier_word): #

    # Sampling
    #def sample(self, temp):

    def sample_word_types(self, temp):
        '''Sample the word types for morpheme segmentation.'''
        # Avoid key issues
        word_type_list = list(self.seen_word_seg.word_seg.keys())
        word_type_list.sort()
        assert (set(word_type_list) ==
        set(self.restaurant.restaurant.keys())), ('All the seen words are not '
        'in the restaurant.')
        print('Resample')
        #print(f'Word list: {word_type_list}')
        #print(f'Segmentation dictionary: {self.seen_word_seg.word_seg}')
        for word_type in word_type_list:
            print('\nword_type', word_type)
            assert (word_type in self.restaurant.restaurant), (
            f'{word_type} not in restaurant')
            morpheme_boundaries = self.seen_word_seg.word_seg[word_type]
            word = SimpleAltHierarchicalWord(word_type, morpheme_boundaries)
            # Remove the word (and morpheme)
            #print(word.morpheme_list)
            #print(f'Segmentation boundaries: {morpheme_boundaries}')
            #print(f'Segmentation boundaries: {word.line_boundaries}')
            remove_word  = self.seen_word_seg.remove_word_and_seg(word_type,
                                morpheme_boundaries)
            #print(f'Remove word: {remove_word}')
            #print(f'Word restaurant: {self.restaurant.restaurant}')
            #print(f'Morpheme restaurant: {self.restaurant_m.restaurant}')
            if remove_word:
                print(f'Removed word: {word.sentence}, {word.line_boundaries}')
                word.remove_morphemes(self.restaurant_m)
            # Sampling
            word.sample_word(self, temp)
            word.morpheme_list = word.decompose()
            #print(f'New decompostion: {word.morpheme_list}')
            # Add the word (and morpheme)
            add_word = self.seen_word_seg.add_word_and_seg(
                        word.sentence, word.line_boundaries)
            #print(f'New segmentation: {word.line_boundaries}')
            #print(f'Add word: {add_word}\n')
            # Add the selected morphemes in the morpheme restaurant
            if add_word:
                print(f'Added word: {word.sentence}, {word.line_boundaries}')
                #self.morpheme_update[self.morpheme_update_index] += 1 # Count updates
                word.add_morphemes(self.restaurant_m)
        print(f'New segmentation dictionary: {self.seen_word_seg.word_seg}')

    #def get_segmented(self):

    #def get_segmented_morph(self):

    #def get_two_level_segmentation(self):


# Utterance in unigram case
class SimpleAltHierUtterance(AltHierUtterance):
    # Information on one utterance of the document
    #def __init__(self, sentence, p_segment, seed=42):


    #def init_boundary(self): # Random case only

    #def init_morph_boundary(self): # Random case only

    #def random_morph_boundary_for_word(self, word): #

    #def numer_base(self, hier_word, state):

    #def left_word(self, i):
    #def right_word(self, i):
    #def centre_word(self, i):

    def find_left_right_centre_words(self, i):
        '''Find the left, right, and centre words with morpheme boundaries.'''
        self.prev = self.prev_boundary(i)
        self.next = self.next_boundary(i)
        # For easier notations
        prev_index = self.prev + 1
        next_index = self.next + 1
        ###utils.check_value_between(i, 0, self.sentence_length - 1) # No last pos###
        left = self.sentence[prev_index:(i + 1)]
        right = self.sentence[(i + 1):next_index]
        centre = self.sentence[prev_index:next_index]
        #[(self.prev + 1):(self.next + 1)]
        # Morpheme boundaries
        left_m_boundaries = self.morph_boundaries[prev_index:(i + 1)]# Check values
        #(not i + 1 beacuse of final True)
        left_m_boundaries.append(True)
        right_m_boundaries = self.morph_boundaries[(i + 1):next_index] # Check values
        centre_m_boundaries = self.morph_boundaries[prev_index:next_index]
        #[(self.prev + 1):(self.next + 1)] # Check
        self.left_word = SimpleAltHierarchicalWord(left, left_m_boundaries)
        self.right_word = SimpleAltHierarchicalWord(right, right_m_boundaries)
        self.centre_word = SimpleAltHierarchicalWord(centre, centre_m_boundaries)

    #def sample_morphemes_in_words(self, state, temp):

    #def update_morph_boundaries(self, new_boundaries):

    #def add_left_and_right(self, state): #

    #def add_centre(self, state): #

    #def sample(self, state, temp):

    #def sample_one(self, i, state, temp): #

    #def prev_boundary(self, i):
    #def next_boundary(self, i):


# Word in unigram case
class SimpleAltHierarchicalWord(AltHierarchicalWord):
    # Information on one utterance of the document
    #def __init__(self, sentence, boundaries): #p_segment):
        #self.sentence = sentence # Unsegmented utterance # Char
        #self.sentence_length = len(self.sentence)
        #self.line_boundaries = boundaries # Here, morpheme-level boundaries
        #self.morpheme_list = self.decompose()


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

    #def sample_one_morph(self, i, state, temp):

    def sample_word(self, state, temp):
        '''Sample the word itself to get the morphemes.'''
        #state.morpheme_scratch[state.morpheme_update_index] += 1 # All sampled
        for i in range(self.sentence_length - 1):
            self.sample_one_morph(i, state, temp)

    #def prev_boundary(self, i):
    #def next_boundary(self, i):

    #def decompose(self):

    #def remove_morphemes(self, restaurant):

    #def add_morphemes(self, restaurant):
