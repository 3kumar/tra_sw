# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 11:55:47 2016

@author: fox
"""

import mdp
import Oger

def generate_internal_weights(reservoir_size, spectral_radius=1, proba=0.2, seed=1, verbose=False):
    """
    Method that generate the weight matrix that will be used for the internal connections of the Reservoir.

    Inputs :

        - verbose: print in the console detailed information.
        - seed: if not None, set the seed of the numpy.random generator to the given value.
        - randomize_seed_afterwards: as the module mdp.numx.random may not be used only by this method,
            the user may want to run several experiments with the same seed only for this method
            (generating the internal weights of the Reservoir), but have random seed for all other
            methods that will use mdp.numx.random.
    """

    mdp.numx.random.seed(seed)
    mask = 1*(mdp.numx_rand.random((reservoir_size,reservoir_size)) < proba)
    mat = mdp.numx.random.normal(0, 1, (reservoir_size,reservoir_size)) #equivalent to mdp.numx.random.randn(n, m) * sd + mu
    w = mdp.numx.multiply(mat, mask)
    if verbose:
        print "Spectral radius of generated matrix before applying another spectral radius: "+str(Oger.utils.get_spectral_radius(w))
    if spectral_radius is not None:
        w *= spectral_radius / Oger.utils.get_spectral_radius(w)
        if verbose:
            print "Spectra radius matrix after applying another spectral radius: "+str(Oger.utils.get_spectral_radius(w))
    return w

def generate_input_weights(reservoir_size, input_dim, input_scaling=1, proba=0.1, seed=1, verbose=False):
    """
    Method that generate the weight matrix that will be used for the input connections of the Reservoir.
    """
    mdp.numx.random.seed(seed)
    mask = 1*(mdp.numx_rand.random((reservoir_size, input_dim))<proba)
    mat = mdp.numx.random.normal(0, 1, (reservoir_size, input_dim))
    w = mdp.numx.multiply(mat, mask)
    if input_scaling is not None:
        w = input_scaling * w

    return w