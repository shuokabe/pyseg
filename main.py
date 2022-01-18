import argparse
import logging
import numpy as np
import pickle
import random
from tqdm import tqdm # Progress bar

#from pyseg import dpseg
from pyseg.dpseg import State
from pyseg.pypseg import PYPState
from pyseg.supervised_dpseg import SupervisionHelper, SupervisedState
from pyseg.supervised_pypseg import SupervisedPYPState
from pyseg.online import online_learning
from pyseg.analysis import Statistics, evaluate, get_boundaries
from pyseg.hyperparameter import Concentration_sampling, Hyperparameter_sampling
from pyseg.nhpylm import NHPYLMState
from pyseg.two_level import TwoLevelState, HierarchicalTwoLevelState
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='path to file')
    parser.add_argument('-m', '--model', default='dpseg', type=str,
                        choices=['dpseg', 'pypseg', 'nhpylm', 'two_level',
                        'htl'],  help='model name')
    parser.add_argument('-a', '--alpha_1', default=20, type=int,
                        help='concentration parameter for unigram DP')
    parser.add_argument('-b', '--p_boundary', default=0.5, type=float,
                        help='prior probability of word boundary')
    parser.add_argument('-d', '--discount', default=0.5, type=float,
                        help='discount parameter for PYP model')
    parser.add_argument('-i', '--iterations', default=100, type=int,
                        help='number of iterations')
    parser.add_argument('-o', '--output_file_base', default='output', type=str,
                        help='output filename (base)')
    parser.add_argument('-r', '--rnd_seed', default=42, type=int,
                        help='random seed')
    parser.add_argument('-s', '--sample_hyperparameter', default=False,
                        type=bool, help='hyperparameter sampling (dp & pypseg)')
    parser.add_argument('-a2', '--alpha_2', default=-1, type=int,
                        help='concentration parameter for bigram model')
    parser.add_argument('-p', '--poisson_parameter', default=0, type=int,
                        help='parameter for NHPYLM Poisson correction')
    parser.add_argument('-am', '--alpha_m', default=20, type=int,
                        help='concentration parameter for morphemes (HTL)')
    parser.add_argument('-dm', '--discount_m', default=0.5, type=float,
                        help='discount parameter for morphemes (HTL)')
    parser.add_argument('-v', '--verbose', default=False,
                        type=bool, help='verbosity of the output')

    # Supervision parameter arguments
    parser.add_argument('--supervision_file', default='none', type=str,
                        help='file name of the data used for supervision')
    parser.add_argument('--supervision_method', default='none', type=str,
                        choices=['none', 'naive', 'naive_freq', 'mixture',
                        'mixture_bigram', 'initialise', 'init_bigram',
                        'init_trigram'],
                        help='supervision method (word dictionary)')
    parser.add_argument('--supervision_parameter', default=0, type=float,
                        help='parameter value for (dictionary) supervision')
    parser.add_argument('--supervision_boundary', default='none', type=str,
                        choices=['none', 'true', 'random', 'sentence', 'word',
                        'morpheme'],
                        help='boundary supervision method')
    parser.add_argument('--supervision_boundary_parameter', default=0, type=float,
                        help='parameter value for boundary supervision')
    parser.add_argument('--online', default='none', type=str,
                        choices=['none', 'without', 'with', 'bigram'],
                        help='online learning method')
    parser.add_argument('--online_batch', default=0, type=int,
                        help='number of sentences after which Gibbs sampling '
                        'is carried out for online learning')
    parser.add_argument('--online_iter', default=0, type=int,
                        help='number of iterations for online learning')

    parser.add_argument('--version', action='version', version='1.5.1')

    return parser.parse_args()


def main():
    args = parse_args()

    # Input file
    filename = args.filename

    # Seed
    rnd_seed = args.rnd_seed
    random.seed(rnd_seed)

    data = open(filename, 'r', encoding = 'utf8').read()
    model_name = args.model

    logging.info(f'Segmenting {filename:s} using {model_name:s}\'s unigram model.')
    logging.info('Boundary initialisation: random') #

    #set_init(b_init) # Copy from segment.cc
    # No set_models since it doesn't seem to be useful for the unigram case

    if (args.supervision_boundary == 'morpheme'): # Two-level segmentation
        raw_data = open(filename, 'r', encoding = 'utf8').read()
        data = utils.morpheme_gold_segment(raw_data, False) # Word level

    supervision = False
    # If supervision
    if (args.supervision_method != 'none') or (args.supervision_boundary != 'none'):
        supervision = True
        if args.supervision_file != 'none': #args.supervision_method != 'none':
            with open(args.supervision_file, 'rb') as d:
                supervision_data = pickle.load(d)
        else:
            supervision_data = 'none'
            #open(args.supervision_file, 'r', encoding = 'utf8').read()
        supervision_helper = SupervisionHelper(supervision_data,
            args.supervision_method, args.supervision_parameter,
            args.supervision_boundary, args.supervision_boundary_parameter)

    # Initialisation of the model state
    if (model_name == 'pypseg') or (args.sample_hyperparameter):
        if model_name == 'dpseg': #args.sample_hyperparameter:
            args.discount = 0
        # If supervision
        if supervision:
            main_state = SupervisedPYPState(
                data, discount = args.discount, alpha_1 = args.alpha_1,
                p_boundary = args.p_boundary, seed = rnd_seed,
                supervision_helper = supervision_helper)
        else:
            main_state = PYPState(data, discount = args.discount,
                               alpha_1 = args.alpha_1,
                               p_boundary = args.p_boundary, seed = rnd_seed)
    elif model_name == 'nhpylm': # NHYPLM model
        #args.sample_hyperparameter = True
        main_state = NHPYLMState(data, alpha_1 = args.alpha_1,
            alpha_2 = args.alpha_2, p_boundary = args.p_boundary,
            poisson_parameter = args.poisson_parameter)
    elif model_name == 'htl': # Hierarchical Two Level model
        args.sample_hyperparameter = True
        args.discount = 0
        main_state = HierarchicalTwoLevelState(data, discount = args.discount,
            alpha_1 = args.alpha_1, p_boundary = args.p_boundary,
            discount_m = args.discount_m, alpha_m = args.alpha_m, seed = rnd_seed)
        # Built-in hyperparameter sampling
    elif model_name == 'two_level':
        args.discount = 0
        # If supervision
        if supervision:
            pass
        else:
            supervision_helper=None
        main_state = TwoLevelState(data, discount = args.discount,
            alpha_1 = args.alpha_1, p_boundary = args.p_boundary,
            seed = rnd_seed, supervision_helper = supervision_helper)
        # Built-in hyperparameter sampling
    else: # Default model: dpseg
        # If supervision
        if supervision:
            main_state = SupervisedState(
                data, alpha_1 = args.alpha_1,
                p_boundary = args.p_boundary, seed = rnd_seed,
                supervision_helper = supervision_helper)
        else:
            main_state = State(data, alpha_1 = args.alpha_1,
                               p_boundary = args.p_boundary)

    iters = args.iterations
    logging.info(f'Sampling {iters:d} iterations.')

    # Hyperparameter sampling initialisation
    hyp_sample = args.sample_hyperparameter
    if hyp_sample:
        logging.info(' Hyperparameter sampled after each iteration.')
        dpseg = bool(model_name == 'dpseg') # dpseg or pypseg model?
        # For dpseg only
        #alpha_sample = Concentration_sampling((1, 1), rnd_seed)
        # For both dpseg and pypseg
        hyperparam_sample = Hyperparameter_sampling((1, 1), (1, 1),
                                                    rnd_seed, dpseg)
        if model_name == 'htl':
            morph_hyper_sample = Hyperparameter_sampling((1, 1), (1, 1),
                rnd_seed, dpseg, morph=True)
    if args.online != 'none':
        logging.info(f'Online learning {args.online} update')
        if (args.online_batch > 0) and (args.online_iter > 0):
            logging.info(f' Every {args.online_batch} sentences, '
                         f'{args.online_iter} iterations.')

    logging.info('Evaluating a sample')

    logging.info(f'Random seed = {rnd_seed:d}')
    logging.info(f'Alphabet size = {main_state.alphabet_size:d}')

    temp_incr = 10 # How many increments of temperature to get to T = 1
    if (iters < temp_incr):
        temp_incr = iters
    iter_incr = iters / temp_incr # Raise temp each iter_incr iters

    # List of temp_incr temperatures (floats)
    temperatures = np.linspace(0.1, 1, temp_incr)
    logging.info(f'Raising temperature in {temp_incr:d} increments'
                 f': {temperatures}')

    # Begin sampling loop
    temp_index = 0
    temp = temperatures[temp_index]
    logging.info(f'iter 0: temp = {temp:.1f}')
    for i in tqdm(range(1, iters + 1)):
        if ((i % iter_incr) == 0) and (i != iters): # Change temperature
            temp_index += 1
            temp = temperatures[temp_index]
            logging.info(f'iter {i:d}: temp = {temp:.1f}')
            if hyp_sample:
                print(f'Current value of alpha: {main_state.alpha_1:.1f}')
                if model_name == 'pypseg':
                    print(f'Current value of d: {main_state.discount:.3f}')
                else:
                    pass

        main_state.sample(temp)
        if model_name not in ['two_level', 'htl']: #model_name != 'two_level':
            utils.check_n_type_token(main_state, args)
        else:
            pass
        # Hyperparameter sampling
        if hyp_sample:
            #main_state.alpha_1 = alpha_sample.sample_concentration(main_state)
            main_state.alpha_1, main_state.discount = \
                    hyperparam_sample.sample_hyperparameter(main_state)
            if model_name == 'htl':
                main_state.alpha_m, main_state.discount_m = \
                        morph_hyper_sample.sample_hyperparameter(main_state)

    if hyp_sample:
        logging.debug(f'Final value of alpha: {main_state.alpha_1:.1f}')
        if model_name == 'pypseg':
            logging.debug(f'Final value of d: {main_state.discount:.3f}')
        elif model_name == 'htl':
            logging.debug(f'Final value of alpha: {main_state.alpha_m:.1f}'
                          '(morpheme)')
        else:
            pass

    # Two-level segmentation
    if (args.supervision_boundary == 'morpheme'): # Two-level segmentation
        raw_data = open(filename, 'r', encoding = 'utf8').read()
        data = utils.morpheme_gold_segment(raw_data, True) # Morpheme level
    # Online learning
    if args.online != 'none':
        loss_list = online_learning(data, main_state, args, temp)

    logging.info(f'{iters:d} iterations')
    if model_name == 'pypseg':
        logging.debug(f'{main_state.restaurant.n_tables} tables')
        #print('Restaurant', main_state.restaurant.restaurant)

    if args.verbose:
        if args.supervision_method in ['naive', 'naive_freq']:
            if (model_name == 'pypseg') or (args.sample_hyperparameter):
                print(main_state.restaurant.restaurant)
            else:
                print(main_state.word_counts.lexicon)
    else:
        pass

    if model_name == 'nhpylm':
        segmented_text_list = [' '.join(utt.word_list)
                               for utt in main_state.utterances]
        segmented_text = '\n'.join(segmented_text_list)
        segmented_text.replace('$', '')
    elif model_name in ['two_level', 'htl']: #model_name == 'two_level':
        if model_name == 'two_level':
            word_segmented_text = main_state.word_state.get_segmented()
            morph_segmented_text = main_state.morph_state.get_segmented()
        else:
            word_segmented_text = main_state.get_segmented()
            morph_segmented_text = main_state.get_segmented_morph()
        stats_word = Statistics(word_segmented_text)
        stats_morph = Statistics(morph_segmented_text)
        #print('State and stats:', main_state.word_counts.lexicon == stats.lexicon)
        logging.info('Word statistics: %s' % (stats_word.stats))
        logging.info('Morph statistics: %s' % (stats_morph.stats))

        word_results = evaluate(main_state.data_word, word_segmented_text)
        logging.info('Word evaluation metrics: %s' % word_results)
        morph_results = evaluate(main_state.data_morph, morph_segmented_text)
        logging.info('Morph evaluation metrics: %s' % morph_results)
        if model_name == 'two_level':
            segmented_text = main_state.get_segmented()
        else:
            segmented_text = main_state.get_two_level_segmentation()
        # Output file (log + segmented text)
        output_file = args.output_file_base + '.txt'
        with open(output_file, 'w',  encoding = 'utf8') as out_text:
            log_info = open('pyseg.log', 'r', encoding = 'utf8').read()
            out_text.write(log_info)
            out_text.write('Segmented text:\n')
            out_text.write(segmented_text)
            if args.online != 'none':
                out_text.write('\nLoss list:\n')
                out_text.write(str(loss_list))
        return 0
    else:
        segmented_text = main_state.get_segmented()

    # Statistics
    stats = Statistics(segmented_text)
    #print('State and stats:', main_state.word_counts.lexicon == stats.lexicon)
    logging.info('Statistics: %s' % (stats.stats))

    # Evaluation results
    results = evaluate(data, segmented_text)
    logging.info('Evaluation metrics: %s' % results)

    # For boundary supervision with segmented sentences
    if args.supervision_boundary == 'sentence':
        logging.info('Without the given sentences:')
        split_gold = utils.text_to_line(data)
        split_seg = utils.text_to_line(segmented_text)
        if args.supervision_boundary_parameter < 1: # Ratio case
            supervision_index = int(np.ceil(
                args.supervision_boundary_parameter * len(split_seg)))
            print('Supervision index:', supervision_index)
        else: # Index case
            supervision_index = int(args.supervision_boundary_parameter)
        #supervision_index = int(args.supervision_boundary_parameter)
        remain_gold = '\n'.join(split_gold[supervision_index:]) + '\n'
        remain_seg = '\n'.join(split_seg[supervision_index:]) + '\n'
        remain_stats = Statistics(remain_seg)
        logging.info(' Remaining statistics: %s' % (remain_stats.stats))
        remain_results = evaluate(remain_gold, remain_seg)
        logging.info(' Remaining evaluation metrics: %s' % (remain_results))


    # Output file (log + segmented text)
    output_file = args.output_file_base + '.txt'
    with open(output_file, 'w',  encoding = 'utf8') as out_text:
        log_info = open('pyseg.log', 'r', encoding = 'utf8').read()
        out_text.write(log_info)
        out_text.write('Segmented text:\n')
        out_text.write(segmented_text)
        if args.online != 'none':
            out_text.write('\nLoss list:\n')
            out_text.write(str(loss_list))


if __name__ == "__main__":
    main()
