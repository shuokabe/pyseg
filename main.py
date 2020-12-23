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
                    filename = 'pyseg.log',
                    filemode = 'w',
                    format = '[%(asctime)s] %(message)s',
                    datefmt = '%d/%m/%Y %H:%M:%S')

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
                        choices=['none', 'boundary', 'naive', 'naive_freq', 'mixture', 'initialise', 'init_bigram'],
                        help='supervision method (boundary and dictionary)')
    parser.add_argument('--supervision_parameter', default=0, type=float,
                        help='parameter value for supervision')

    parser.add_argument('--version', action='version', version='1.1.0')

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

    logging.info('Segmenting {0:s} using {1:s}\'s unigram model.'.format(filename, model_name))
    logging.info('Boundary initialisation: random') #

    #set_init(b_init) # Copy from segment.cc
    # No set_models since it doesn't seem to be useful for the unigram case

    # Initialisation of the model state
    if model_name == 'pypseg':
        main_state = PYPState(data, discount = args.discount,
                           alpha_1 = args.alpha_1, p_boundary = args.p_boundary)
    else: # Default model: dpseg
        if args.supervision_method != 'none': # If supervision
            if args.supervision_method != 'boundary':
                with open(args.supervision_file, 'rb') as d:
                    supervision_data = pickle.load(d)
            else:
                supervision_data = open(args.supervision_file, 'r', encoding = 'utf8').read()
            main_state = SupervisedState(data, alpha_1 = args.alpha_1,
                         p_boundary = args.p_boundary,
                         supervision_data = supervision_data,
                         supervision_method = args.supervision_method,
                         supervision_parameter = args.supervision_parameter)
        else:
            main_state = State(data, alpha_1 = args.alpha_1,
                         p_boundary = args.p_boundary)

    iters = args.iterations
    logging.info('Sampling {:d} iterations.'.format(iters))

    logging.info('Evaluating a sample')

    logging.info('Random seed = {:d}'.format(rnd_seed))
    logging.info('Alphabet size = {:d}'.format(main_state.alphabet_size))

    temp_incr = 10 # How many increments of temperature to get to T = 1
    if (iters < temp_incr):
        temp_incr = iters
    iter_incr = iters / temp_incr # Raise temp each iter_incr iters

    # List of temp_incr temperatures (floats)
    temperatures = np.linspace(0.1, 1, temp_incr)
    logging.info('Raising temperature in {0:d} increments: {1}'.format(temp_incr, temperatures))

    # Begin sampling loop
    temp_index = 0
    temp = temperatures[temp_index]
    logging.info('iter 0: temp = {:.1f}'.format(temp))
    for i in tqdm(range(1, iters + 1)):
        if ((i % iter_incr) == 0) and (i != iters): # Change temperature
            temp_index += 1
            temp = temperatures[temp_index]
            logging.info('iter {0:d}: temp = {1:.1f}'.format(i, temp))

        main_state.sample(temp)
        if args.supervision_method in ['naive', 'naive_freq']:
            pass
        else:
            utils.check_equality(main_state.word_counts.n_types, len(main_state.word_counts.lexicon))
            utils.check_equality(main_state.word_counts.n_tokens, sum(main_state.word_counts.lexicon.values()))

    logging.info('{:d} iterations'.format(iters))
    if model_name == 'pypseg':
        utils.check_value_between(main_state.restaurant.n_tables, main_state.word_counts.n_types, main_state.word_counts.n_tokens)
        utils.check_equality((sum(main_state.restaurant.customers.values())), main_state.word_counts.n_tokens)
        utils.check_equality(main_state.restaurant.n_customers, main_state.word_counts.n_tokens)
        print('{} tables'.format(main_state.restaurant.n_tables))
        #print('Restaurant', main_state.restaurant.restaurant)

    segmented_text = main_state.get_segmented()

    output_file = args.output_file_base + '.txt'
    with open(output_file, 'w',  encoding = 'utf8') as out_text:
        out_text.write(segmented_text)

    # Statistics
    stats = Statistics(segmented_text)
    #print('State and stats:', main_state.word_counts.lexicon == stats.lexicon)
    logging.info('Statistics: %s' % (stats.stats))

    # Evaluation results
    results = evaluate(data, segmented_text)
    logging.info('Evaluation metrics: %s' % results)


if __name__ == "__main__":
    main()
