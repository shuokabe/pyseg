import argparse
import logging
import numpy as np
import pickle
import random
from tqdm import tqdm # Progress bar

#from pyseg import dpseg
from pyseg.dpseg import State
from pyseg.pypseg import PYPState
from pyseg.supervised_dpseg import SupervisedState
from pyseg.analysis import Statistics, evaluate
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
                        choices=['dpseg', 'pypseg'], help='model name')
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
                        choices=['none', 'true', 'random', 'sentence'],
                        help='boundary supervision method')
    parser.add_argument('--supervision_boundary_parameter', default=0, type=float,
                        help='parameter value for boundary supervision')

    parser.add_argument('--version', action='version', version='1.2.7')

    return parser.parse_args()


def main():
    args = parse_args()

    # Input file
    filename = args.filename

    # Seed
    rnd_seed = args.rnd_seed #42
    random.seed(rnd_seed)

    data = open(filename, 'r', encoding = 'utf8').read()
    model_name = args.model

    logging.info(f'Segmenting {filename:s} using {model_name:s}\'s unigram model.')
    logging.info('Boundary initialisation: random') #

    #set_init(b_init) # Copy from segment.cc
    # No set_models since it doesn't seem to be useful for the unigram case

    # Initialisation of the model state
    if model_name == 'pypseg':
        main_state = PYPState(data, discount = args.discount,
                           alpha_1 = args.alpha_1, p_boundary = args.p_boundary,
                           seed = rnd_seed)
    else: # Default model: dpseg
        if (args.supervision_method != 'none') or (args.supervision_boundary != 'none'): # If supervision
            if args.supervision_method != 'none':
                with open(args.supervision_file, 'rb') as d:
                    supervision_data = pickle.load(d)
            else:
                supervision_data = 'none'
                #open(args.supervision_file, 'r', encoding = 'utf8').read()
            main_state = SupervisedState(data, alpha_1 = args.alpha_1,
                         p_boundary = args.p_boundary, seed = rnd_seed,
                         supervision_data = supervision_data,
                         supervision_method = args.supervision_method,
                         supervision_parameter = args.supervision_parameter,
                         supervision_boundary = args.supervision_boundary,
                         supervision_boundary_parameter = args.supervision_boundary_parameter)
        else:
            main_state = State(data, alpha_1 = args.alpha_1,
                         p_boundary = args.p_boundary)

    iters = args.iterations
    logging.info(f'Sampling {iters:d} iterations.')

    logging.info('Evaluating a sample')

    logging.info(f'Random seed = {rnd_seed:d}')
    logging.info('Alphabet size = {:d}'.format(main_state.alphabet_size))

    temp_incr = 10 # How many increments of temperature to get to T = 1
    if (iters < temp_incr):
        temp_incr = iters
    iter_incr = iters / temp_incr # Raise temp each iter_incr iters

    # List of temp_incr temperatures (floats)
    temperatures = np.linspace(0.1, 1, temp_incr)
    logging.info(f'Raising temperature in {temp_incr:d} increments: {temperatures}')

    # Begin sampling loop
    temp_index = 0
    temp = temperatures[temp_index]
    logging.info(f'iter 0: temp = {temp:.1f}')
    for i in tqdm(range(1, iters + 1)):
        if ((i % iter_incr) == 0) and (i != iters): # Change temperature
            temp_index += 1
            temp = temperatures[temp_index]
            logging.info(f'iter {i:d}: temp = {temp:.1f}') #.format(i, temp))

        main_state.sample(temp)
        if args.supervision_method in ['naive', 'naive_freq']:
            pass
        elif model_name == 'pypseg':
            utils.check_equality(main_state.restaurant.n_customers, sum(main_state.restaurant.customers.values()))
            utils.check_equality(main_state.restaurant.n_tables, sum(main_state.restaurant.tables.values()))
        else:
            utils.check_equality(main_state.word_counts.n_types, len(main_state.word_counts.lexicon))
            utils.check_equality(main_state.word_counts.n_tokens,
                                 sum(main_state.word_counts.lexicon.values()))

    logging.info(f'{iters:d} iterations')
    if model_name == 'pypseg':
        #utils.check_value_between(main_state.restaurant.n_tables,
        #                          main_state.word_counts.n_types,
        #                          main_state.word_counts.n_tokens)
        #utils.check_equality((sum(main_state.restaurant.customers.values())),
        #                      main_state.word_counts.n_tokens)
        #utils.check_equality(main_state.restaurant.n_customers, main_state.word_counts.n_tokens)
        logging.debug('{} tables'.format(main_state.restaurant.n_tables))
        #print('Restaurant', main_state.restaurant.restaurant)

    segmented_text = main_state.get_segmented()

    #output_file = args.output_file_base + '.txt'
    #with open(output_file, 'w',  encoding = 'utf8') as out_text:
    #    out_text.write(segmented_text)

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
        split_gold = utils.text_to_line(data, True)
        split_seg = utils.text_to_line(segmented_text, True)
        supervision_index = int(args.supervision_boundary_parameter)
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


if __name__ == "__main__":
    main()
