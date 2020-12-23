import collections
import logging
import random

from scipy.stats import poisson

from pyseg import utils

# Supervised dpseg model

logging.basicConfig(level = logging.DEBUG, format = '[%(asctime)s] %(message)s',
                    datefmt = '%d/%m/%Y %H:%M:%S')

from pyseg.dpseg import Lexicon, State, Utterance


# Unigram case
class SupervisedState(State): # Information on the whole document
    def __init__(self, data, alpha_1, p_boundary, supervision_data=None,
                 supervision_method='none', supervision_parameter=0):
        # State parameters
        self.alpha_1 = alpha_1
        self.p_boundary = p_boundary
        utils.check_probability(p_boundary)

        self.beta = 2 # Hyperparameter?

        logging.info(' alpha_1: {0:d}, p_boundary: {1:.1f}'.format(self.alpha_1, self.p_boundary))

        # Supervision variable
        self.sup_data = supervision_data # dictionary or text file
        self.sup_method = supervision_method
        self.sup_parameter = supervision_parameter

        if self.sup_method == 'boundary':
            logging.info('Supervision with segmentation boundaries')
        else: # Dictionary supervision
            logging.info('Supervision with a dictionary')
        logging.info(' Supervision method: {0:s}, supervision parameter: {1:.2f}'.format(self.sup_method, self.sup_parameter))

        # Data and Utterance object
        self.unsegmented = utils.unsegmented(data) #datafile.unsegmented(data)
        self.unsegmented_list = utils.text_to_line(self.unsegmented, True) # Remove empty string
        #self.unsegmented_list = utils.delete_value_from_vector(self.unsegmented_list, '') # Remove empty string

        # Variable to store alphabet, utterance, and lexicon information
        self.utterances = [] #? # Stored utterances
        self.boundaries = [] # In case a boundary element is needed

        if self.sup_method == 'boundary':
            if self.unsegmented != utils.unsegmented(self.sup_data):
                raise ValueError('The supervision data must have the same content as the input text.')
            else:
                pass
            self.sup_boundaries = [] # Stored supervision boundaries
            sup_data_list = utils.text_to_line(self.sup_data, True)
            utils.check_equality(len(self.unsegmented_list), len(sup_data_list))
            for i in range(len(self.unsegmented_list)): # rewrite with correct variable names
                unseg_line = self.unsegmented_list[i]
                sup_line = sup_data_list[i]
                utterance = SupervisedUtterance(unseg_line, sup_line, p_boundary)
                self.utterances.append(utterance)
                self.boundaries.append(utterance.line_boundaries)
                self.sup_boundaries.append(utterance.sup_boundaries)
        else: # Dictionary supervision (or no supervision) case
            for unseg_line in self.unsegmented_list: # rewrite with correct variable names
                utterance = Utterance(unseg_line, p_boundary)
                self.utterances += [utterance]
                self.boundaries += [utterance.line_boundaries]

        self.n_utterances = len(self.utterances) # Number of utterances

        # Lexicon object (Counter)
        self.word_counts = Lexicon() # Word counter
        init_segmented_list = utils.text_to_line(self.get_segmented(), True) # Remove empty string
        #init_segmented_list = utils.delete_value_from_vector(init_segmented_list, '')
        self.word_counts.init_lexicon_text(init_segmented_list)

        if self.sup_method == 'naive':
            naive_dictionary = {word: self.sup_parameter for word, frequency in self.sup_data.items()}
            self.word_counts.lexicon = self.word_counts.lexicon + collections.Counter(naive_dictionary)
            print('Naive dictionary', self.word_counts.lexicon)

        if self.sup_method == 'naive_freq':
            naive_dictionary = {word: frequency * self.sup_parameter for word, frequency in self.sup_data.items()}
            self.word_counts.lexicon = self.word_counts.lexicon + collections.Counter(naive_dictionary)
            print('Naive freq dictionary', self.word_counts.lexicon)

        # Alphabet (list of letters)
        self.alphabet = utils.delete_value_from_vector(list(set(self.unsegmented)), '\n')
        self.alphabet_size = len(self.alphabet)

        # Phoneme probability (dictionary)
        self.phoneme_ps = dict()

        if self.sup_method in ['initialise', 'init_bigram']:
            self.word_length_ps = dict() # Exclusive to the dictionary initialise method

        #self.init_probs() # How to initialise the boundaries (here: random)
        self.init_phoneme_probs()



    def init_phoneme_probs(self):
        '''
        Computes (uniform distribution)
        ### TODO: complete the documentation
        '''
        # Skip part to calculate the true distribution of characters

        if self.sup_method in ['initialise', 'init_bigram']:
            # Supervision with a dictionary
            logging.info('Phoneme distribution: dictionary supervision')
            if self.sup_method == 'init_bigram':
                chosen_method = 'bigram'
            else:
                chosen_method = 'empirical'
            logging.info(' Chosen initialisation method: {0:s}'.format(chosen_method))

            words_in_dict_str = ''
            word_length_dict = dict()
            for word, frequency in self.sup_data.items():
                words_in_dict_str += word #* frequency # For letter probabilities
                word_length = len(word)
                word_length_dict.setdefault(word_length, 0)
                word_length_dict[word_length] += 1 #= word_length_dict.get(len(word), 0) + 1 #frequency # For length probabilities
            print('words in dict_str:', words_in_dict_str[0:50])
            total_frequence = sum(word_length_dict.values())
            mean_token_length = sum([word_length * frequency for word_length, frequency in word_length_dict.items()]) / total_frequence
            print('mean TL:', mean_token_length)
            if (chosen_method == 'empirical'):
                self.word_length_ps = {word_length: frequency / total_frequence for word_length, frequency in word_length_dict.items()}
            else:
                self.word_length_ps = {i: poisson.pmf(i, mean_token_length, loc=1) for i in range(max(word_length_dict.keys()))}
            #self.word_length_ps = {word_length: frequency / total_frequence for word_length, frequency in word_length_dict.items()}
            print('word_length:', self.word_length_ps, sum(self.word_length_ps.values()))

            # TODO: make the different cases clearer (and more efficient)
            if chosen_method == 'bigram':
                bigrams_in_dict_list = []
                for word in self.sup_data.keys():
                    considered_word = '<{0}>'.format(word)
                    word_bigram_list = [considered_word[i:(i + 2)] for i in range(len(considered_word) - 1)]
                    bigrams_in_dict_list += word_bigram_list
                letters_in_dict = collections.Counter(bigrams_in_dict_list)
                frequency_bigrams_dict = sum(letters_in_dict.values())
                for bigram in letters_in_dict.keys():
                    self.phoneme_ps[bigram] = letters_in_dict[bigram] / frequency_bigrams_dict
                print('Bigram dictionary: {0}'.format(self.phoneme_ps))
            else:
                letters_in_dict = collections.Counter(words_in_dict_str)
            frequency_letters_dict = sum(letters_in_dict.values())

            # TODO: deal with the case letters_in_dict[letter] == 0
            if chosen_method == 'length':
                self.phoneme_ps = {letter: 1 / self.alphabet_size for letter in self.alphabet}
            elif chosen_method == 'bigram':
                pass
            else:
                self.phoneme_ps = {letter: letters_in_dict[letter] / frequency_letters_dict for letter in self.alphabet}
            #for letter in self.alphabet:
                #elif chosen_method == 'bigram':
                    #pass
                #else:
                    #self.phoneme_ps[letter] = letters_in_dict[letter] / frequency_letters_dict
            #assert (abs(sum(self.phoneme_ps.values()) - 1.0) < 10^(-5)), 'The sum of the probabilities is not 1.'
            print('Sum of probabilities: {0}'.format(sum(self.phoneme_ps.values())))

        else:
            # Uniform distribution case
            logging.info('Phoneme distribution: uniform')

            #self.phoneme_ps = {letter: 1 / self.alphabet_size for letter in self.alphabet}
            for letter in self.alphabet:
                self.phoneme_ps[letter] = 1 / self.alphabet_size

    # Probabilities
    #def p_cont(self):

    def p_word(self, string):
        '''
        Computes the prior probability of a string of length n:
        p_word = alpha_1 * p_boundary * (1 - p_boundary)^(n - 1)
                * \prod_1^n phoneme(string_i)
        '''
        p = 1
        if (self.sup_method == 'init_bigram'):
            considered_word = '<{0}>'.format(string)
            for i in range(len(considered_word) - 1):
                bigram = considered_word[i:(i + 2)]
                p = p * self.phoneme_ps.get(bigram, 10 ** (-6))
        else:
            for letter in string:
                p = p * self.phoneme_ps[letter]
        p = p * ((1 - self.p_boundary) ** (len(string) - 1)) * self.p_boundary
        if self.sup_method == 'mixture':
            #print('p before mixture:', p)
            n_words_dict = sum(self.sup_data.values())
            p = self.sup_parameter / n_words_dict * utils.indicator(string, self.sup_data) \
                + (1 - self.sup_parameter) * p
            #print('p after mixture:', p)
        elif self.sup_method in ['initialise', 'init_bigram']:
            #print('p before length:', p)
            p = p * self.word_length_ps.get(len(string), 10 ** (-5))
            #print('p after length:', p)
        else:
            pass
        return p * self.alpha_1

    # Sampling
    #def sample(self, temp):

    #def get_segmented(self):

# Utterance in unigram case
class SupervisedUtterance(Utterance): # Information on one utterance of the document
    def __init__(self, sentence, sup_sentence, p_segment):
        self.sentence = sentence # Unsegmented utterance str
        self.sup_sentence = sup_sentence # Supervision sentence (with spaces)
        self.p_segment = p_segment
        utils.check_probability(p_segment)

        self.line_boundaries = [] # Test to store boundary existence
        self.init_boundary() #

        self.sup_boundaries = []
        self.get_sup_boundaries()

        utils.check_equality(len(self.sentence), len(self.sup_boundaries))

    #def init_boundary(self):

    def get_sup_boundaries(self):
        boundary_track = 0
        unseg_length = len(self.sentence)
        for i in range(unseg_length - 1):
            if self.sup_sentence[boundary_track + 1] == ' ':
                self.sup_boundaries.append(True)
                boundary_track += 1
            else:
                self.sup_boundaries.append(False)
            boundary_track += 1
        self.sup_boundaries.append(True)

    #def numer_base(self, word, state):

    #def left_word(self, i):

    #def right_word(self, i):

    #def centre_word(self, i):

    #def sample(self, state, temp):

    def sample_one(self, i, state, temp):
        lexicon = state.word_counts
        left = self.left_word(i)
        right = self.right_word(i)
        centre = self.centre_word(i)
        ### boundaries is the boundary for the utterance only here
        if self.line_boundaries[i]: # Boundary at the i-th position ('yes' case)
            #print('yes case')
            lexicon.remove_one(left) #lexicon[left] = lexicon[left] - 1
            lexicon.remove_one(right)
            #print(left, lexicon.lexicon[left], right, lexicon.lexicon[right])
        else: # No boundary at the i-th position ('no' case)
            #print('no case')
            lexicon.remove_one(centre)
            #print(centre, lexicon.lexicon[centre])

        if self.sup_boundaries[i]:
            # No sampling if known boundary
            self.line_boundaries[i] = True
            lexicon.add_one(left)
            lexicon.add_one(right)
        else:
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

    #def prev_boundary(self, i):

    #def next_boundary(self, i):
