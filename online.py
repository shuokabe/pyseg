import argparse
import logging
#import numpy as np
#import pickle
import random
from tqdm import tqdm # Progress bar
import types

#from pyseg import dpseg
from pyseg.dpseg import State
from pyseg.pypseg import PYPState
from pyseg.supervised_dpseg import SupervisionHelper, SupervisedState
from pyseg.supervised_pypseg import SupervisedPYPState
from pyseg.analysis import Statistics, evaluate, get_boundaries
from pyseg.hyperparameter import Hyperparameter_sampling
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
#def loss(gold, segmented):
#    '''Loss function to compare boundaries during online learning.'''
#    n = len(gold) # The last boundary is always True
#    utils.check_equality(n + 1, len(segmented))
#    loss = 0
#    for i in range(n):
#        loss += (gold[i] - segmented[i]) ** 2
#    return loss / n

def count_correction(gold, segmented):
    '''Loss function to compare boundaries during online learning.'''
    n = len(gold) # The last boundary is always True
    utils.check_equality(n + 1, len(segmented))
    count = 0
    for i in range(n):
        count += (gold[i] - segmented[i]) ** 2
    return count #/ n

def get_segmented_sentence(sentence, boundaries):
    '''Get the segmented sentence according to the boundaries.'''
    segmented_line = []
    beg = 0
    pos = 0
    utils.check_equality(len(sentence), len(boundaries))
    for boundary in boundaries:
        if boundary == 1: # If there is a boundary
            segmented_line.append(sentence[beg:(pos + 1)])
            beg = pos + 1
        pos += 1
    return segmented_line

def bigram_p_word(state, string): # p_bigram_character_model
    '''Bigram p_word function for online learning'''
    p = 1
    # Character model
    considered_word = f'<{string:s}>'
    for i in range(len(considered_word) - 1):
        ngram = considered_word[i:(i + 2)]
        p = p * state.phoneme_ps[ngram]
    return p

def dynamic_sup_boundaries(utt, gold_boundary, cache):
    '''Supervision boundaries when using dynamic cache and online learning.'''
    sent_word_list = get_segmented_sentence(utt.sentence, utt.sup_boundaries)
    full_gold_boundary = gold_boundary + [1]
    gold_seg_list = get_segmented_sentence(utt.sentence, full_gold_boundary)
    new_sup_boundaries = []
    j = 0
    for word in sent_word_list:
        word_length = len(word)
        if word in cache: # Use the stored boundaries for the word
            utils.check_equality(len(cache[word]), word_length)
            new_sup_boundaries.extend(cache[word])
        else:
            new_sup_boundaries.extend([-1] * (word_length - 1))
            new_sup_boundaries.append(1)
            cache[word] = full_gold_boundary[j:(j + word_length)]
        j += word_length
    return new_sup_boundaries

class OnlineLearningHelper:
    '''Helper object for online learning (mainly parameters).

    Parameters
    ----------
    args : ArgumentParser
        Arguments parsed from the commands (see main.py).

    Attributes
    ----------

    '''
    def __init__(self, args):
        self.model_name = args.model
        self.online = args.online
        self.batch = args.online_batch
        self.iter = args.online_iter # Number of iterations
        self.hyp_sample = args.sample_hyperparameter # Hyperparameter sampling

        # Bools
        self.update = bool(self.online in ['with', 'bigram'])
        self.bigram = bool(self.online == 'bigram') # Bigram character model?

        logging.info('Online learning:')


def online_learning(data, state, args, temp):
    '''Online learning function'''
    #logging.info('Online learning:')
    on = OnlineLearningHelper(args)
    #model_name = args.model
    dynamic = False
    # Online learning parameters
    online = args.online
    if dynamic:
        #online = 'bigram'
        on.bigram = True
        utils.check_equality(args.supervision_boundary, 'morpheme')
        logging.info(' Using dynamic cache')
    # Every on_batch sentences, Gibbs sampling on the remaining text.
    on_batch = args.online_batch
    on_iter = args.online_iter # Number of iterations
    # Hyperparameter sampling initialisation
    #hyp_sample = args.sample_hyperparameter
    if on.hyp_sample:
        dpseg = bool(on.model_name == 'dpseg') # dpseg or pypseg model? #
        hyperparam_sample = Hyperparameter_sampling((1, 1), (1, 1),
                                                    args.rnd_seed, dpseg)
    split_gold = utils.text_to_line(data)
    gold_boundaries = get_boundaries(split_gold)
    loss_list = []
    if on.bigram:
        sup_dictionary = dict()
    if dynamic:
        dynamic_cache = dict()
    update_incr = state.n_utterances // 10
    #for i in tqdm(range(1, iters + 1)):
    for i in tqdm(range(state.n_utterances)):
        gold = gold_boundaries[i]
        utterance = state.utterances[i]
        if dynamic:
            utterance.sup_boundaries = dynamic_sup_boundaries(
                                            utterance, gold, dynamic_cache)
        utterance.sample(state, temp)
        segmented = utterance.line_boundaries
        l = count_correction(gold, segmented)
        #print(l)
        loss_list.append(l)
        if on.update: # Update the dictionary
            unsegmented_line = utterance.sentence
            segmented_line = get_segmented_sentence(unsegmented_line, segmented)
            #print('segmented_line', segmented_line)
            gold_line = utils.line_to_word(split_gold[i])
            #print('gold line', gold_line)
            if (on.model_name == 'pypseg') or on.hyp_sample: #(args.sample_hyperparameter): #
                for word in segmented_line:
                    state.restaurant.remove_customer(word)
                for word in gold_line:
                    state.restaurant.add_customer(word)
                    if on.bigram:
                        sup_dictionary[word] = sup_dictionary.get(word, 0) + 1
            else:
                for word in segmented_line:
                    state.word_counts.remove_one(word)
                for word in gold_line:
                    state.word_counts.add_one(word)
                    if on.bigram:
                        sup_dictionary[word] = sup_dictionary.get(word, 0) + 1
            # For bigram online learning
            if ((i + 1) % update_incr == 0) and (on.bigram):
                #print(i, sup_dictionary)
                sup = SupervisionHelper(sup_dictionary, 'none', 'none', 'none',
                                        'none')
                state.phoneme_ps = sup.set_bigram_character_model(state.alphabet)
                print(f'Sum of probabilities: {sum(state.phoneme_ps.values())}')
                changeFunction = types.MethodType
                state.p_word = changeFunction(bigram_p_word, state)
                print(f'p_word test: {state.p_word("test")}')
            ## Test for on_batch and on_iter
            if (on.batch > 0) and ((i % on.batch) == 0) and (on.iter > 0):
                for iter_count in range(on.iter):
                    #print(iter_count)
                    for j in range(i + 1, state.n_utterances):
                        state.utterances[j].sample(state, temp)
                    utils.check_n_type_token(state, args)
                    # Hyperparameter sampling
                    if on.hyp_sample:
                        state.alpha_1, state.discount = \
                            hyperparam_sample.sample_hyperparameter(state)
        else:
            pass
    if on.hyp_sample:
        logging.debug(f'Final value of alpha: {state.alpha_1:.1f}')
        if on.model_name == 'pypseg':#
            logging.debug(f'Final value of d: {state.discount:.3f}')
    return loss_list
