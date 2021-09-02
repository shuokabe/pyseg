import math
import random

#from numpy.random import default_rng
from numpy.random import RandomState
from scipy.stats import beta, gamma


class Concentration_sampling:
    def __init__(self, prior, seed=42):
        '''Object to ease sampling the concentration parameter.

        Parameters
        ----------
        a_prior : float
            Gamma prior (shpae parameter) for the concentration parameter
        b_prior : float
            Gamma prior (scale parameter) for the concentration parameter
        seed : integer
            Seed value to generate random values (exclusive to this object)

        Attributes
        ----------
        a_prior : float
            Gamma prior (shpae parameter) for the concentration parameter
        b_prior : float
            Gamma prior (scale parameter) for the concentration parameter
        random_gen : default_rng()
            Random number generator for the scipy random variables

        '''
        self.a_prior, self.b_prior = prior
        #self.random_gen = default_rng(seed)
        self.random_gen = RandomState(seed) # Legacy Random Generation

    def sample_concentration(self, state):
        '''Sampling the concentration hyperparameter alpha.'''
        #n, k = state.word_counts.n_tokens, state.word_counts.n_types
        n, k = state.restaurant.n_customers, state.restaurant.n_tables
        # Auxiliary varaible eta
        eta = self.random_gen.beta(state.alpha_1 + 1, n)
        #eta = beta.rvs(state.alpha_1 + 1, n, random_state = self.random_gen)
        new_a = self.a_prior + k - 1 # Shape parameter
        new_b = self.b_prior - math.log(eta) # Scale parameter
        pi_eta = new_a / (new_a + (n * new_b))
        gamma_1 = self.random_gen.gamma(self.a_prior + k, new_b)
        #gamma_1 = gamma.rvs(self.a_prior + k, scale = new_b, random_state = self.random_gen)
        gamma_2 = self.random_gen.gamma(new_a, new_b)
        #gamma_2 = gamma.rvs(new_a, scale = new_b, random_state = self.random_gen)
        new_alpha = pi_eta * gamma_1 + (1 - pi_eta) * gamma_2
        return new_alpha
