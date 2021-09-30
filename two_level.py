import logging

# Two-level model (word and morpheme)

logging.basicConfig(level = logging.DEBUG, format = '[%(asctime)s] %(message)s',
                    datefmt = '%d/%m/%Y %H:%M:%S')

from pyseg.dpseg import State
from pyseg.pypseg import PYPState
from pyseg.supervised_dpseg import SupervisionHelper, SupervisedState
from pyseg.hyperparameter import Hyperparameter_sampling
from pyseg import utils


class TwoLevelState(State):
    def __init__(self, raw_data, discount, alpha_1, p_boundary, seed=42,
                 supervision_helper=None):
        # raw_data is a text segmented in morphemes with '-'
        self.data_word = utils.morpheme_gold_segment(raw_data, False) # Word level
        self.data_morph = utils.morpheme_gold_segment(raw_data, True) # Morpheme level

        # Two states for two levels
        logging.info('Word level model:')
        self.word_state = PYPState(self.data_word, discount = discount,
                                alpha_1 = alpha_1, p_boundary = p_boundary,
                                seed = seed)
        logging.info('Morpheme level model:')
        self.morph_state = PYPState(self.data_morph, discount = discount,
                                    alpha_1 = alpha_1, p_boundary = p_boundary,
                                    seed = seed)
        ### Harmonise the initial state

        self.n_utterances = len(self.word_state.utterances)
        utils.check_equality(len(self.morph_state.utterances), self.n_utterances)

        self.alphabet_size = self.word_state.alphabet_size # For main.py

        self.word_hyper_sample = Hyperparameter_sampling((1, 1), (1, 1), seed, True)
        self.morph_hyper_sample = Hyperparameter_sampling((1, 1), (1, 1), seed, True)

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
                utt_word.sample_one(i, self.word_state, temp)
                if utt_word.line_boundaries[i] == True: # i is a word boundary
                    self.put_boundary(i, self.morph_state, utt_morph)
                else: # i is not a word boundary
                    utt_morph.sample_one(i, self.morph_state, temp)

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
