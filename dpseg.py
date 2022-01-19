import collections
import logging
import random

from pyseg import utils

# dpseg model

logging.basicConfig(level = logging.DEBUG, format = '[%(asctime)s] %(message)s',
                    datefmt = '%d/%m/%Y %H:%M:%S')


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

        # Number of types and tokens (size parameters)
        self.n_types = 0
        self.n_tokens = 0

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
        self.update_lex_size()


# Unigram case
class State: # Information on the whole document
    def __init__(self, data, alpha_1, p_boundary):
        # State parameters
        self.alpha_1 = alpha_1
        self.p_boundary = p_boundary
        utils.check_probability(self.p_boundary)

        self.beta = 2 # Hyperparameter?

        logging.info(f' alpha_1: {self.alpha_1:d}, p_boundary: {self.p_boundary:.1f}')

        # Data and Utterance object
        self.unsegmented = utils.unsegmented(data)
        # Remove empty string
        self.unsegmented_list = utils.text_to_line(self.unsegmented)

        # Variable to store alphabet, utterance, and lexicon information
        self.utterances = [] # Stored Utterance objects

        for unseg_line in self.unsegmented_list: # rewrite with correct variable names
            # do next_reference function
            utterance = Utterance(unseg_line, self.p_boundary)
            self.utterances.append(utterance)

        self.n_utterances = len(self.utterances) # Number of utterances

        # Lexicon object (Counter)
        self.word_counts = Lexicon() # Word counter
        # Remove empty string
        init_segmented_list = utils.text_to_line(self.get_segmented())
        self.word_counts.init_lexicon_text(init_segmented_list)

        # Alphabet (list of letters)
        self.alphabet = utils.delete_value_from_vector(list(set(self.unsegmented)), '\n')
        self.alphabet_size = len(self.alphabet)

        # Phoneme probability (dictionary)
        self.phoneme_ps = dict()
        #self.init_probs() # How to initialise the boundaries (here: random)
        self.init_phoneme_probs()


    # Initialisation
    #def init_probs(self): #
        #for u in self.utterances:
            #words = u.get_reference_words() #
            #for w in words:
                #true_words_ps += [w] # unknown
        # Normalise: give empirical prob over unigrams
        #for w in true_words_ps:
            #ntokens = true_words_ps.ntokens()
        #self.init_phoneme_probs()

    def init_phoneme_probs(self):
        '''
        Computes (uniform distribution)
        ### to complete
        '''
        # Skip part to calculate the true distribution of characters

        # Uniform distribution case
        logging.info('Phoneme distribution: uniform')

        for letter in self.alphabet:
            self.phoneme_ps[letter] = 1 / self.alphabet_size

    # Probabilities
    def p_cont(self):
        n_words = self.word_counts.n_tokens
        p = (n_words - self.n_utterances + 1 + self.beta / 2) / (n_words + 1 + self.beta)
        utils.check_probability(p)
        return p

    def p_word(self, string):
        '''
        Computes the prior probability of a string of length n:
        p_word = alpha_1 * p_boundary * (1 - p_boundary)^(n - 1)
                * \prod_1^n phoneme(string_i)
        '''
        p = 1
        for letter in string:
            p = p * self.phoneme_ps[letter]
        p = p * ((1 - self.p_boundary) ** (len(string) - 1)) * self.p_boundary
        return p * self.alpha_1

    # Sampling
    def sample(self, temp):
        #word_counts.check_invariant() #
        # The input text is the unsegmented text in a list type

        utils.check_equality(len(self.utterances), self.n_utterances) # See if needed
        for utterance in self.utterances: #
            #print('Utterance: ', utterance, 'boundary: ', boundary_utt)
            utterance.sample(self, temp) # model_type = 1

    def get_segmented(self):
        '''Generate the segmented text with the current state of the boundaries.

        ###
        '''
        segmented_text_list = []
        utils.check_equality(len(self.utterances), len(self.unsegmented_list))
        for i in range(len(self.utterances)):
            #segmented_line_list = []
            unsegmented_line = self.unsegmented_list[i]
            boundaries_line = self.utterances[i].line_boundaries
            #beg = 0
            #pos = 0
            #utils.check_equality(len(boundaries_line), len(unsegmented_line))
            ##utils.check_equality(len(self.boundaries[i]), len(unsegmented_line))
            #for boundary in boundaries_line:
            #    if boundary: # If there is a boundary
            #        segmented_line_list += [unsegmented_line[beg:(pos + 1)]]
            #        beg = pos + 1
            #    pos += 1
            segmented_line_list = utils.segment_sentence_with_boundaries(
                unsegmented_line, boundaries_line)
            # Convert list of words into a string sentence
            segmented_line = ' '.join(segmented_line_list)
            segmented_text_list.append(segmented_line)
        return '\n'.join(segmented_text_list) #segmented_text


# Utterance in unigram case
class Utterance: # Information on one utterance of the document
    '''Utterance class for each sentence of the document.

    Parameters
    ----------
    sentence : string
        Unsegmented sentence
    p_segment : float
        Probability to have a boundary at a given position

    Attributes
    ----------
    line_boundaries : list of bools [bools] (or [0 or 1])
        List of boundary statuses (0 or 1) for each position in the sentence.

    '''
    def __init__(self, sentence, p_segment):
        self.sentence = sentence # Unsegmented utterance
        self.p_segment = p_segment
        utils.check_probability(p_segment)
        self.sentence_length = len(self.sentence)

        self.line_boundaries = []
        self.init_boundary()

    def init_boundary(self): # Random case only
        for i in range(len(self.sentence) - 1):
            rand_val = random.random()
            if rand_val < self.p_segment:
                self.line_boundaries.append(True)
            else:
                self.line_boundaries.append(False)
        self.line_boundaries.append(True)

    def numer_base(self, word, state):
        '''Return the numerator from the fraction used in sampling.'''
        return state.word_counts.lexicon[word] + state.p_word(word)

    def left_word(self, i):
        '''Return the word on the left of i.'''
        utils.check_value_between(i, 0, self.sentence_length) #len(self.sentence))
        prev = self.prev_boundary(i)
        return self.sentence[(prev + 1):(i + 1)]

    def right_word(self, i):
        '''Return the word on the right of i.'''
        utils.check_value_between(i, 0, self.sentence_length - 1) # No last pos
        next = self.next_boundary(i)
        return self.sentence[(i + 1):(next + 1)]

    def centre_word(self, i):
        '''Return the word which contains the i-th position in the sentence.'''
        utils.check_value_between(i, 0, self.sentence_length - 1) # No last pos
        prev = self.prev_boundary(i)
        next = self.next_boundary(i)
        return self.sentence[(prev + 1):(next + 1)]

    def sample(self, state, temp):
        #if (model_type == 1): # Unigram model
        # Final boundary posn must always be true, so don't sample it. #
        #print('Utterance: ', self.sentence, 'boundary: ', self.line_boundaries)
        utils.check_equality(len(self.line_boundaries), self.sentence_length)

        for i in range(self.sentence_length - 1):
            self.sample_one(i, state, temp)

    def sample_one(self, i, state, temp):
        lexicon = state.word_counts
        left = self.left_word(i)
        right = self.right_word(i)
        centre = self.centre_word(i)
        ### boundaries is the boundary for the utterance only here
        if self.line_boundaries[i]: # Boundary at the i-th position ('yes' case)
            #print('yes case')
            lexicon.remove_one(left)
            lexicon.remove_one(right)
            #print(left, lexicon.lexicon[left], right, lexicon.lexicon[right])
        else: # No boundary at the i-th position ('no' case)
            #print('no case')
            lexicon.remove_one(centre)
            #print(centre, lexicon.lexicon[centre])

        denom = lexicon.n_tokens + state.alpha_1
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
        no = no ** temp
        p_yes = yes / (yes + no)
        if (random.random() < p_yes):
            #print('Boundary case')
            self.line_boundaries[i] = True
            lexicon.add_one(left)
            lexicon.add_one(right)
        else:
            #print('No boundary case')
            self.line_boundaries[i] = False
            lexicon.add_one(centre)

    #def log_posterior(self, n_utterances, lexicon, state): # Seems not to intervene
    #    pass

    def prev_boundary(self, i):
        '''Return the index of the previous boundary with respect to i.'''
        utils.check_value_between(i, 0, self.sentence_length)
        for j in range(i - 1, -1, -1):
            if self.line_boundaries[j] == True:
                return j
        return -1

    def next_boundary(self, i):
        '''Return the index of the next boundary with respect to i.'''
        utils.check_value_between(i, 0, self.sentence_length - 1) # No last pos
        # Start search from (i + 1)
        return self.line_boundaries.index(True, i + 1)
