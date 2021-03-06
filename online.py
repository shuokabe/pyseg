import argparse
import logging
#import numpy as np
#import pickle
import random
from tqdm import tqdm # Progress bar

#from pyseg import dpseg
from pyseg.dpseg import State
from pyseg.pypseg import PYPState
from pyseg.supervised_dpseg import SupervisionHelper, SupervisedState
from pyseg.supervised_pypseg import SupervisedPYPState
from pyseg.analysis import Statistics, evaluate, get_boundaries
from pyseg import utils

# General setup of libraries
logging.basicConfig(level = logging.DEBUG,
                    #filename = 'pyseg.log',
                    #filemode = 'w',
                    format = '[%(asctime)s] %(message)s',
                    datefmt = '%d/%m/%Y %H:%M:%S')

filelog = logging.FileHandler('pyseg.log', 'w')
filelog.setLevel(logging.INFO)

formatter = logging.Formatter(fmt = '[%(asctime)s] %(message)s',
                              datefmt = '%d/%m/%Y %H:%M:%S')
filelog.setFormatter(formatter)
logging.getLogger().addHandler(filelog)

# Corresponds to the segment file

# Online learning
def loss(gold, segmented):
    '''Loss function to compare boundaries during online learning.'''
    n = len(gold) # The last boundary is always True
    utils.check_equality(n + 1, len(segmented))
    loss = 0
    for i in range(n):
        loss += (gold[i] - segmented[i]) ** 2
    return loss / n

def count_correction(gold, segmented):
    '''Loss function to compare boundaries during online learning.'''
    n = len(gold) # The last boundary is always True
    utils.check_equality(n + 1, len(segmented))
    count = 0
    for i in range(n):
        count += (gold[i] - segmented[i]) ** 2
    return count

def get_segmented_sentence(sentence, boundaries):
    '''Get the segmented sentence according to the boundaries.'''
    segmented_line = []
    beg = 0
    pos = 0
    utils.check_equality(len(sentence), len(boundaries))
    for boundary in boundaries:
        if boundary: # If there is a boundary
            segmented_line.append(sentence[beg:(pos + 1)])
            beg = pos + 1
        pos += 1
    return segmented_line

def bigram_p_word(state, string):
    '''Bigram p_word function for online learning'''
    p = 1
    # Character model
    considered_word = f'<{string:s}>'
    for i in range(len(considered_word) - 1):
        ngram = considered_word[i:(i + 2)]
        p = p * state.phoneme_ps[ngram]
    return p

def online_learning(data, state, args, temp):
    '''Online learning function'''
    import types
    model_name = args.model
    # Online learning parameters
    online = args.online
    # Every on_batch sentences, Gibbs sampling on the remaining text.
    on_batch = args.online_batch
    on_iter = args.online_iter # Number of iterations
    if online in ['with', 'bigram']:
        update = True
    else:
        update = False
    split_gold = utils.text_to_line(data)
    gold_boundaries = get_boundaries(split_gold)
    loss_list = []
    if online == 'bigram':
        sup_dictionary = dict()
    update_incr = state.n_utterances / 10
    #for i in tqdm(range(1, iters + 1)):
    logging.info('Online learning:')
    for i in tqdm(range(state.n_utterances)):
        gold = gold_boundaries[i]
        utterance = state.utterances[i]
        utterance.sample(state, temp)
        segmented = utterance.line_boundaries
        #l = loss(gold, segmented)
        l = count_correction(gold, segmented)
        #print(l)
        loss_list.append(l)
        if update: # Update the dictionary
            unsegmented_line = utterance.sentence
            segmented_line = get_segmented_sentence(unsegmented_line, segmented)
            #print('segmented_line', segmented_line)
            gold_line = utils.line_to_word(split_gold[i])
            #print('gold line', gold_line)
            if model_name == 'pypseg':
                for word in segmented_line:
                    state.restaurant.remove_customer(word)
                for word in gold_line:
                    state.restaurant.add_customer(word)
                    if online == 'bigram':
                        sup_dictionary[word] = sup_dictionary.get(word, 0) + 1
            else:
                for word in segmented_line:
                    state.word_counts.remove_one(word)
                for word in gold_line:
                    state.word_counts.add_one(word)
                    if online == 'bigram':
                        sup_dictionary[word] = sup_dictionary.get(word, 0) + 1
            # For bigram online learning
            if (i % update_incr == 0) and (online == 'bigram'):
                #print(i, sup_dictionary)
                sup = SupervisionHelper(sup_dictionary, 'none', 'none', 'none',
                                        'none')
                state.phoneme_ps = sup.set_bigram_character_model(state.alphabet)
                print(f'Sum of probabilities: {sum(state.phoneme_ps.values())}')
                changeFunction = types.MethodType
                state.p_word = changeFunction(bigram_p_word, state)
                print(f'p_word test: {state.p_word("test")}')
            ## Test for on_batch and on_iter
            if (on_batch > 0) and ((i % on_batch) == 0) and (on_iter > 0):
                for iter_count in range(on_iter):
                    #print(iter_count)
                    for j in range(i + 1, state.n_utterances):
                        state.utterances[j].sample(state, temp)
                    utils.check_n_type_token(state, args)
        else:
            pass
    return loss_list
