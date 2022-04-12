import numpy as np
import math
import random

#from numpy.random import default_rng
from numpy.random import RandomState
from scipy.stats import beta, gamma

from pyseg import utils


class Concentration_sampling:
    def __init__(self, prior, seed=42):
        '''Object to ease sampling the concentration parameter.

        Parameters
        ----------
        prior : pair of float
            Gamma prior (shape, scale) for the concentration parameter
        seed : integer
            Seed value to generate random values (exclusive to this object)

        Attributes
        ----------
        a_prior : float
            Gamma prior (shape parameter) for the concentration parameter
        b_prior : float
            Gamma prior (scale parameter) for the concentration parameter
        random_gen : default_rng()
            Random number generator for the scipy random variables

        '''
        self.a_prior, self.b_prior = prior
        #self.random_gen = default_rng(seed)
        self.random_gen = RandomState(seed) # Legacy Random Generation

    def sample_concentration(self, state):
        '''Sample the concentration parameter alpha.'''
        #n, k = state.word_counts.n_tokens, state.word_counts.n_types
        n, k = state.restaurant.n_customers, state.restaurant.n_tables
        # Auxiliary varaible eta
        eta = self.random_gen.beta(state.alpha_1 + 1, n)
        #eta = beta.rvs(state.alpha_1 + 1, n, random_state = self.random_gen)
        new_a = self.a_prior + k - 1 # Shape parameter
        new_b = self.b_prior - math.log(eta) # Scale parameter
        pi_eta = new_a / (new_a + (n * new_b))
        gamma_1 = self.random_gen.gamma(self.a_prior + k, new_b)
        #gamma_1 = gamma.rvs(self.a_prior + k, scale = new_b,
        #        random_state = self.random_gen)
        gamma_2 = self.random_gen.gamma(new_a, new_b)
        #gamma_2 = gamma.rvs(new_a, scale = new_b, random_state = self.random_gen)
        new_alpha = pi_eta * gamma_1 + (1 - pi_eta) * gamma_2
        return new_alpha


class Hyperparameter_sampling:
    def __init__(self, theta_prior, d_prior, seed=42, dpseg=False, morph=False):
        '''Object to ease hyperparameter sampling.

        Hyperparameters: concentration parameter and discount parameter.
        The notations are the same as in Teh (2006).
        Parameters
        ----------
        theta_prior : pair of float
            Gamma prior (shape, scale) for the concentration parameter
        d_prior : pair of float
            Beta prior (a, b) for the discount parameter
        seed : integer
            Seed value to generate random values (exclusive to this object)
        dpseg : bool
            Bool to indicate the model type (dpseg or pypseg)
        morph : bool
            Bool to indicate if morpheme hyperparameters are sampled (for HTL)

        Attributes
        ----------
        alpha_prior : float
            Gamma prior (shape parameter) for the concentration parameter (theta)
        beta_prior : float
            Gamma prior (scale parameter) for the concentration parameter (theta)
        a_prior : float
            Beta prior (a) for the discount parameter (d)
        b_prior : float
            Beta prior (b) for the discount parameter (d)
        random_gen : default_rng()
            Random number generator for the scipy random variables
        dpseg : bool
            Bool to indicate the model type (dpseg or pypseg)
        morph : bool
            Bool to indicate if morpheme hyperparameters are sampled (for HTL)

        '''
        self.alpha_prior, self.beta_prior = theta_prior
        self.a_prior, self.b_prior = d_prior
        #self.random_gen = default_rng(seed)
        self.random_gen = RandomState(seed) # Legacy Random Generation
        self.dpseg = dpseg
        self.morph = morph

    def sample_hyperparameter(self, state):
        '''Sample the hyperparameters.'''
        self.sample_auxiliary(state)
        concentration = self.sample_concentration()
        if self.dpseg: # dpseg model, i.e. discount == 0
            return concentration, 0
        else: # Other models
            discount = self.sample_discount()
            return concentration, discount

    def sample_concentration(self):
        '''Sample the concentration parameter alpha (here called theta).'''
        Y = sum(self.y_i_list)
        return self.random_gen.gamma(self.alpha_prior + Y,
                                     self.beta_prior - np.log(self.x))

    def sample_discount(self):
        '''Sample the discount parameter d.'''
        Y = sum([1 - y for y in self.y_i_list])
        Z = sum([1 - z for z in self.z_wkj_list])
        return self.random_gen.beta(self.a_prior + Y, self.b_prior + Z)

    def sample_auxiliary(self, state):
        '''Sample auxiliary variable x, y_i, z_wkj.'''
        if self.morph: # For morpheme-level hyperparameters in the HTL model
            restaurant = state.restaurant_m
        else: # Other models (default)
            restaurant = state.restaurant
        n, t = restaurant.n_customers, restaurant.n_tables
        # x
        if n <= 1:
            self.x = 1
        else:
            self.x = self.random_gen.beta(state.alpha_1 + 1, n - 1)
        # y_i
        self.y_i_list = []
        for i in range(1, t): # if t == 1: y_i_list = []
            y_i = state.alpha_1 / (state.alpha_1 + state.discount * i)
            utils.check_probability(y_i)
            self.y_i_list.append(self.random_gen.binomial(1, y_i))
        utils.check_equality(len(self.y_i_list), t - 1)
        # z_wkj
        self.z_wkj_list = []
        for table_w_list in restaurant.restaurant.values():
            # Check each table with word label w
            for c_wk in table_w_list:
                if c_wk >= 2:
                    for j in range(1, int(c_wk)):
                        z_wkj = (j - 1) / (j - state.discount)
                        utils.check_probability(z_wkj)
                        self.z_wkj_list.append(self.random_gen.binomial(1, z_wkj))
                else:
                    pass
