import numpy as np
import random
import logging
import argparse
from tqdm import tqdm # Progress bar

#from pyseg import dpseg
from pyseg.dpseg import State
from pyseg import utils

# General setup of libraries
logging.basicConfig(level = logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')

# Corresponds to the segment file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='Path to file')
    parser.add_argument('-a', '--alpha_1', default=20, type=int,
                        help='total unigram generator weight')
    parser.add_argument('-b', '--p_boundary', default=0.5, type=float,
                        help='prior probability of word boundary')
    parser.add_argument('-i', '--iterations', default=100, type=int,
                        help='number of iterations')
    parser.add_argument('-r', '--rnd_seed', default=42, type=int,
                        help='random seed')

    return parser.parse_args()


def main():

    args = parse_args()

    # Input file
    filename = args.filename #'../Summer exp 2020/mboshi_0.5_letter.word' #

    # Seed
    rnd_seed = args.rnd_seed #42 #
    random.seed(rnd_seed)

    data = open(filename, 'r').read()  # Add the preprocessing function

    logging.info('Segmenting {:s} using unigram model.'.format(filename))
    logging.info('Boundary initialisation: random') #

    #set_init(b_init) # Copy from segment.cc
    # No set_models since it doesn't seem to be useful with just the unigram case
    main_state = State(data, alpha_1 = args.alpha_1, p_boundary = args.p_boundary) #

    iters = args.iterations #
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
    logging.info('{:d} iterations'.format(iters))

    segmented_pydpseg = utils.text_to_line(main_state.get_segmented())

    with open('pydpseg_test.txt', 'w') as out_text:
        out_text.write('\n'.join(segmented_pydpseg))

if __name__ == "__main__":
    main()
