from collections import Counter

from pyseg import utils

# Corresponds to the scoring files and the wordseg statistics file. #


class Statistics:
    '''Statistics class containing statistics about the given text.

    Parameters
    ----------
    text : string
        Already segmented text (gold and model output)

    Attributes
    ----------
    split_text : list of strings [string]
        List containing all the sentences of the given text.
    split_utterances : list of list of strings [[string]]
        List containing the list of each sentence's (list) words.
    lexicon : Counter()
        List of the words in the text with their frequencies.
    stats : dictionary {statistics: value}
        Dictionary with the five statistics given by the class:
        - N_utterances: number of utterances
        - N_tokens: number of tokens
        - avg_token_length: average length of tokens (WL)
        - N_types: number of types
        - avg_type_length: average length of word types (TL)

    '''
    def __init__(self, text):
        # Text is an already segmented text (string)
        self.split_text = utils.text_to_line(text)
        #self.split_text = utils.delete_value_from_vector(self.split_text, '')
        self.split_utterances = [utils.line_to_word(line)
                                 for line in self.split_text]

        self.lexicon = self.lexicon_text()
        self.stats = self.compute_stats()

    #def __str__(self):
    #    str_lexicon = ''
    #    for key, value in self.stats.items():
    #        str_lexicon += '\n' + '{0}: {1}'.format(key, value)
    #    return str_lexicon[:len(str_lexicon)]

    def lexicon_text(self):
        '''Create the lexicon (Counter) of the text.'''
        list_word = []
        for split_line in self.split_utterances:
            list_word += split_line
        return Counter(list_word)

    def average_token_length(self):
        '''Return the average length of tokens.'''
        tl_sum = sum([sum([len(token) for token in utt])
                      for utt in self.split_utterances])
        n_tokens = sum([len(utt) for utt in self.split_utterances])
        return tl_sum / n_tokens

    def average_type_length(self):
        '''Return the average length of types.'''
        type_list = list(self.lexicon)
        return sum([len(word) for word in type_list]) / len(type_list)

    def compute_stats(self):
        '''Create a dictionary containing the statistics for the text.'''
        stats = dict()

        stats['N_utterances'] = len(self.split_text)

        # Length of utterances in number of words (number of words per sentence)
        tokens_len = [len(utt) for utt in self.split_utterances]
        # Number of tokens
        stats['N_tokens'] = sum(tokens_len)

        # Average length of tokens
        stats['avg_token_length'] = self.average_token_length()

        # Number of types
        stats['N_types'] = len(self.lexicon)

        # Average length of types
        stats['avg_type_length'] = self.average_type_length()

        return stats


### Evaluation part ###

def f_measure(p, r):
    return 2 * p * r / (p + r)

def get_boundaries(text_list):
    '''Gets the boundaries (list of 0 and 1) from a (split) text.

    The last boundary is always True, so not considered here.

    Parameters
    ----------
    text_list : list of string [string]
        Text split into sentences (list of sentences)

    Returns
    -------
    boundaries : list of list of integers [[integer]]
        List containing boundary statuses (0 or 1) for each sentence (list).
    '''
    boundaries = []
    for line in text_list:
        line_boundaries = []
        boundary_track = 0
        unseg_length = len(utils.unsegmented(line))
        for i in range(unseg_length - 1):
            if line[boundary_track + 1] == ' ':
                line_boundaries.append(True)
                boundary_track += 1
            else:
                line_boundaries.append(False)
            boundary_track += 1
        boundaries.append(line_boundaries)
    return boundaries

#def boundaries_eval(gold_boundaries, segmented_boundaries):
#    boundary_counts = Counter()
#    for i in range(len(gold_boundaries)):
#        boundary_counts += Counter(zip(gold_boundaries[i], segmented_boundaries[i]))
#    print(boundary_counts)

#    tp = boundary_counts[(True, True)]
#    b_precision = tp / (tp + boundary_counts[(False, True)])
#    b_recall = tp / (tp + boundary_counts[(True, False)])
#    b_f_score = f_measure(b_precision, b_recall)

#    return b_precision, b_recall, b_f_score

def token_count_utt(gold_boundary, segmented_boundary):
    # Count the number of exactly matching tokens (at the boundary level)
    # In one utterance.
    token_match = True
    n_token_match = 0
    for i in range(len(gold_boundary)):
        gold_bool = gold_boundary[i]
        seg_bool = segmented_boundary[i]

        if gold_bool and seg_bool: # Boundaries match
            if token_match: # The current token is in both utterances
                n_token_match += 1
            # Start a new token
            token_match = True

        # Boundaries don't match
        elif (gold_bool and not seg_bool) or (not gold_bool and seg_bool):
            token_match = False
        else: # Non-boundaries match: currently in a token
            continue

    if token_match: # Last token
        n_token_match += 1
    return n_token_match

def evaluate(reference, segmented):
    '''
    Compute the PRF metrics (precision, recall, and F-score) at three levels.

    Both text inputs are raw strings.
    PRF metrics at boundary (B), token (W), and type level (L).

    Parameters
    ----------
    reference : string
        Raw (i.e. not split) reference text
    segmented : string
        Raw text to compare (e.g. output from a segmentation model)

    Returns
    -------
    eval : dictionary {metrics: value}
        Dictionary containing all three metrics for the three evaluation levels
    '''
    # Statistics
    reference_stats = Statistics(reference)
    segmented_stats = Statistics(segmented)

    reference_list = utils.text_to_line(reference)
    segmented_list = utils.text_to_line(segmented)
    utils.check_equality(len(reference_list), len(segmented_list))

    ref_boundaries = get_boundaries(reference_list)
    seg_boundaries = get_boundaries(segmented_list)

    boundary_counts = Counter()
    match_token_count = 0
    for i in range(len(ref_boundaries)):
        # For boundary metrics
        boundary_counts += Counter(zip(ref_boundaries[i], seg_boundaries[i]))
        # For token metrics
        match_token_count += token_count_utt(ref_boundaries[i], seg_boundaries[i])
    print('Boundary count', boundary_counts)
    print('Match token count', match_token_count)

    # Boundary metrics (BP, BR, and BF)
    boundary_tp = boundary_counts[(True, True)]
    bp = boundary_tp / (boundary_tp + boundary_counts[(False, True)])
    br = boundary_tp / (boundary_tp + boundary_counts[(True, False)])
    bf = f_measure(bp, br)

    # Token metrics (TP, TR, and TF)
    wp = match_token_count / segmented_stats.stats['N_tokens']
    wr = match_token_count / reference_stats.stats['N_tokens']
    wf = f_measure(wp, wr)

    # Type metrics (LP, LR, and LF)
    ref_type_set = set(reference_stats.lexicon)
    seg_type_set = set(segmented_stats.lexicon)
    lp = len(ref_type_set & seg_type_set) / segmented_stats.stats['N_types']
    lr = len(ref_type_set & seg_type_set) / reference_stats.stats['N_types']
    lf = f_measure(lp, lr)

    eval = dict(
            BP = 100 * bp, BR = 100 * br, BF = 100 * bf,
            WP = 100 * wp, WR = 100 * wr, WF = 100 * wf,
            LP = 100 * lp, LR = 100 * lr, LF = 100 * lf
    )
    return eval
