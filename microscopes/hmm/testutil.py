"""
Test helpers specific to HMM
"""

import numpy as np
from microscopes.common import validator
from microscopes.hmm.definition import model_definition

def toy_model(defn, states=5):
    # generate the observation and transition matrix
    obs_mat   = np.random.dirichlet(alpha=np.ones(defn.N()),size=states)
    trans_mat = np.random.dirichlet(alpha=np.ones(states),size=states)

    return obs_mat, trans_mat

def toy_dataset(defn, states=5, avglen=100, numobs=100):
    """Create a toy dataset for evaluating HMM inference

    Parameters
    ----------
    defn:   model definition
    states: number of latent states
    avlen:  average length of one observation sequence
      (actual length is sampled from a poisson distribution)
    numobs: number of observation sequences
    """

    validator.validate_type(defn, model_definition, 'defn')

    obs_mat, trans_mat = toy_model(defn, states)

    # generate data
    data = []
    for i in xrange(numobs):
      T = np.random.poisson(lam=avglen)
      state = 0
      data.append(T * [None])
      for t in xrange(T):
        state      = np.nonzero(np.random.multinomial(n=1,pvals=trans_mat[state]))[0][0]
        data[i][t] = np.nonzero(np.random.multinomial(n=1,pvals=obs_mat[state]))[0][0]
    return data

def toy_dataset_and_states(defn, states=5, avglen=100, numobs=100):
    """Create a toy dataset for evaluating HMM inference, along with the state sequence

    Parameters
    ----------
    defn:   model definition
    states: number of latent states
    avlen:  average length of one observation sequence
      (actual length is sampled from a poisson distribution)
    numobs: number of observation sequences
    """

    validator.validate_type(defn, model_definition, 'defn')

    obs_mat, trans_mat = toy_model(defn, states)

    # generate data
    data = []
    states = []
    for i in xrange(numobs):
      T = np.random.poisson(lam=avglen)
      state = 0
      data.append(T * [None])
      states.append(T * [None])
      for t in xrange(T):
        state      = np.nonzero(np.random.multinomial(n=1,pvals=trans_mat[state]))[0][0]
        data[i][t] = np.nonzero(np.random.multinomial(n=1,pvals=obs_mat[state]))[0][0]
        states[i][t] = state
    return data, states