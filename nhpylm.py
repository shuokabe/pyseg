import collections
import logging
import numpy as np
import random

from scipy.special import logsumexp
from scipy.stats import poisson

from pyseg import utils

# NHPYLM model from Mochihashi (bigram model)

logging.basicConfig(level = logging.DEBUG, format = '[%(asctime)s] %(message)s',
                    datefmt = '%d/%m/%Y %H:%M:%S')


from pyseg.dpseg import Lexicon, State, Utterance


class Lexicon(Lexicon): # Improved dictionary using a Counter
    #def __init__(self):

    #def update_lex_size(self):

    #def add_one(self, word):

    #def remove_one(self, word):

    #def init_lexicon_text(self, text):

    def init_bigram_lexicon_text(self, text):
        '''Initialises the lexicon (Counter) with bigrams from the text.'''
        #counter = collections.Counter()
        bigram_list = []
        for line in text:
            split_line = utils.line_to_word(line)
            #counter += collections.Counter(split_line) # Keep counter type
            bigram_list.extend(list(zip(split_line, split_line[1:])))
        self.lexicon = collections.Counter(bigram_list) #counter
        self.update_lex_size()


# Unigram case
class NHPYLMState(State): # Information on the whole document
    def __init__(self, data, alpha_1, alpha_2, p_boundary, poisson_parameter):
        # State parameters
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.p_boundary = p_boundary
        utils.check_probability(self.p_boundary)
        self.poisson_parameter = poisson_parameter # Poisson correction

        # Unigram or bigram model
        if self.alpha_2 < 0:
            self.bigram = False
        else:
            self.bigram = True

        self.beta = 2 # Hyperparameter?

        logging.info(f' alpha_1: {self.alpha_1:d}, alpha_2: {self.alpha_2:d}, '
                     f'p_boundary: {self.p_boundary:.1f}, '
                     f'Poisson parameter: {self.poisson_parameter}')

        # Data and Utterance object
        self.unsegmented = utils.unsegmented(data)
        # Remove empty string
        self.unsegmented_list = utils.text_to_line(self.unsegmented)
        # Add EOS mark ($)
        self.unsegmented_list = utils.add_EOS(self.unsegmented_list)

        # Variable to store alphabet, utterance, and lexicon information
        self.utterances = [] # Stored Utterance objects

        for unseg_line in self.unsegmented_list: # rewrite with correct variable names
            # do next_reference function
            utterance = NHPYLMUtterance(unseg_line, self.p_boundary)
            self.utterances.append(utterance)

        self.n_utterances = len(self.utterances) # Number of utterances

        # Lexicon object (Counter)
        self.word_counts = Lexicon() # Word counter
        # Remove empty string
        init_segmented_list = utils.text_to_line(self.get_segmented())
        self.word_counts.init_lexicon_text(init_segmented_list)

        # Bigram restaurant
        self.bigram_counts = Lexicon() # Word counter
        self.bigram_counts.init_bigram_lexicon_text(init_segmented_list)

        # Alphabet (list of letters)
        self.alphabet = utils.delete_value_from_vector(
                        list(set(f'{self.unsegmented}$')), '\n')
        self.alphabet_size = len(self.alphabet)

        # Phoneme probability (dictionary)
        self.phoneme_ps = dict()
        #self.init_probs() # How to initialise the boundaries (here: random)
        self.init_phoneme_probs()

        # Poisson correction #
        if self.poisson_parameter > 0:
            self.poisson = True
            self.poisson_correction() # Length probabilities for Poisson correction

    #def init_phoneme_probs(self):

    def poisson_correction(self):
        '''Length probabilities for Poisson correction.'''
        sentence_length_list = [len(sent.sentence) for sent in self.utterances]
        max_length = max(sentence_length_list)
        mean_token_length = self.poisson_parameter
        self.word_length_ps = {i: poisson.pmf(i, mean_token_length, loc=1)
                               for i in range(max_length)}
        print('word_length:', self.word_length_ps, sum(self.word_length_ps.values()))

    # Probabilities
    #def p_cont(self):

    def p_word(self, string):
        '''
        Computes the prior probability of a string of length n:
        p_word = p_boundary * (1 - p_boundary)^(n - 1)
                * \prod_1^n phoneme(string_i)
        '''
        p = 1
        for letter in string:
            p = p * self.phoneme_ps[letter]
        m = len(string)
        if self.poisson:
            p = p * self.word_length_ps.get(m, 10 ** (-5))
        else:
            p = p * ((1 - self.p_boundary) ** (m - 1)) * self.p_boundary
        return p * self.alpha_1

    def p_unigram(self, word):
        '''Compute bigram probability: p(word).'''
        unigram_denom = self.word_counts.n_tokens + self.alpha_1
        unigram_num = self.word_counts.lexicon[word] + self.p_word(word) #\
        #            + self.alpha_1 * self.p_word(word)
        unigram_prob = unigram_num / unigram_denom
        #print(f'unigram prob of {word}', unigram_prob)
        return unigram_prob

    def p_bigram(self, word, w_before):
        '''Compute bigram probability: p(word|w_before).'''
        bigram_denom = self.word_counts.lexicon[w_before] + self.alpha_2
        unigram_prob = self.p_unigram(word)
        #print('unigram prob', unigram_prob)
        bigram_num = self.bigram_counts.lexicon[(w_before, word)] \
                   + self.alpha_2 * unigram_prob
        #print('bigram_num', bigram_num)
        #print('bigram_denom', bigram_denom)
        return bigram_num / bigram_denom

    # Sampling
    #def sample(self, temp):

    #def get_segmented(self):


class NHPYLMUtterance(Utterance): # Information on one utterance of the document
    def __init__(self, sentence, p_segment):
        self.sentence = sentence # Unsegmented utterance
        self.p_segment = p_segment
        utils.check_probability(p_segment)

        self.line_boundaries = []
        self.init_boundary()

        self.word_list = []
        self.init_word_list()

    #def init_boundary(self): # Random case only

    def init_word_list(self):
        '''Initialise the word list of the sentence with the boundaries.'''
        beg = 0
        pos = 0
        #utils.check_equality(len(boundaries_line), len(unsegmented_line))
        #utils.check_equality(len(self.boundaries[i]), len(unsegmented_line))
        for boundary in self.line_boundaries[:-1]: # No EOS mark
            if boundary: # If there is a boundary
                self.word_list += [self.sentence[beg:(pos + 1)]]
                beg = pos + 1
            pos += 1

    #def numer_base(self, word, state):

    #def left_word(self, i):

    #def right_word(self, i):

    #def centre_word(self, i):

    def sample(self, state, temp=0):
        '''Blocked Gibbs Sampling'''
        #if (model_type == 1): # Unigram model
        # Final boundary posn must always be true, so don't sample it. #
        #print('Utterance: ', self.sentence, 'boundary: ', self.line_boundaries)
        utils.check_equality(len(self.line_boundaries), len(self.sentence))

        ### Remove words in sentence
        bigram_list = utils.bigram_list(self.word_list)
        for word in self.word_list:
            state.word_counts.remove_one(word)
        for bigram in bigram_list:
            state.bigram_counts.remove_one(bigram)

        ### Compute forward variables
        self.forward(state)

        ### Backward sampling
        self.word_list = self.backward(state)
        self.word_list.reverse()

        # Add words in the new sentence
        bigram_list = utils.bigram_list(self.word_list)
        for word in self.word_list:
            state.word_counts.add_one(word) # unigram
        for bigram in bigram_list:
            state.bigram_counts.add_one(bigram) # bigram

    def forward(self, state): # Mochihashi
        '''Forward filtering for blocked Gibbs sampling.'''
        n = len(self.line_boundaries)
        self.forward_alpha = np.ones((n + 1, n + 1)) # forward_alpha[0][0] = 1
        for t in range(1, (n + 1)): # As in the paper
            for k in range(1, (t + 1)):
                self.forward_alpha[t][k] = np.exp(self.log_forward(t, k, state))

    def log_forward(self, t, k, state):
        '''Compute the log of the forward variable (more efficient).'''
        bigram = state.bigram #
        if (t == 0) and (k == 0):
            return 1
        else:
            t_k = t - k
            word = self.sentence[(t_k):t] # +1 -1
            # j = 0 case
            log_forward_list = [np.log(state.p_unigram(word))]
            for j in range(1, (t_k + 1)):
                word_before = self.sentence[(t_k - j):(t_k)] # +1 -1
                #print(f'word: {word}, word_before: {word_before}\n')
                if bigram: #
                    bigram_prob = state.p_bigram(word, word_before)
                    log_forward_list.append(np.log(bigram_prob) \
                            + np.log(self.forward_alpha[t_k][j])) #log_forward(t_k, j)
                else: #
                    unigram_prob = state.p_unigram(word) #
                    log_forward_list.append(np.log(unigram_prob) \
                            + np.log(self.forward_alpha[t_k][j])) #
            return logsumexp(log_forward_list) #logsumexp

    def draw_backward(self, word, t, state):
        '''Draw k from the possible values for backward sampling.'''
        drawn_k = 0
        bigram = state.bigram #
        # Compute the probability for each possible k
        backward_prob_list = []
        for k in range(1, (t + 1)):
            if bigram:
                word_before = self.sentence[(t - k + 1 - 1):t]
                backward_prob = state.p_bigram(word, word_before) \
                                * self.forward_alpha[t][k]
                backward_prob_list.append(backward_prob)
            else:
                backward_prob = state.p_unigram(word) * self.forward_alpha[t][k]
                backward_prob_list.append(backward_prob)
        #print(f'backward_prob_list: {backward_prob_list}')
        # Normalise the probabilities
        prob_sum = sum(backward_prob_list)
        norm_backward_prob = [prob / prob_sum for prob in backward_prob_list]
        #print(f'norm_backward_prob: {norm_backward_prob}')
        # Draw k
        draw_value = random.random()
        cumulative_sum = 0
        for k in range(t):
            cumulative_sum += norm_backward_prob[k]
            if draw_value <= cumulative_sum:
                drawn_k = k + 1 # Check index
                break
            else:
                pass
        if (drawn_k == 0):
            print(f'draw_value: {draw_value}')
        return drawn_k

    def backward(self, state):
        '''Backward sampling for blocked Gibbs sampling.'''
        # Initialisation
        t = len(self.sentence)
        i = 0
        word = '$'
        word_list = []
        while t > 0:
            #print(f'Backward t: {t}')
            # Not argmax but sampling
            k = self.draw_backward(word, t, state)
            word_before = self.sentence[(t - k + 1 - 1):t]
            word = word_before
            t = t - k
            i += 1
            self.line_boundaries[t - 1] = True # Check index
            word_list.append(word)
        return word_list

    #def sample_one(self, i, state, temp):

    #def prev_boundary(self, i):

    #def next_boundary(self, i):
