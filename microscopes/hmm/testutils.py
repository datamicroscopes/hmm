"""
Test helpers specific to HMM
"""

import numpy as np


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

    # generate the observation and transition matrix
    obs_mat   = np.random.dirichlet(alpha=np.ones(defn.N()),size=states)
    trans_mat = np.random.dirichlet(alpha=np.ones(states),size=states)

    # generate data
    data = []
    for i in xrange(numobs):
      T = np.random.poisson(lam=avglen)
      state = 0
      data.append([])
      for t in xrange(T):
        print state
        data[i].append(np.nonzero(np.random.multinomial(n=1,pvals=obs_mat[state]))[0][0])
        state = np.nonzero(np.random.multinomial(n=1,pvals=trans_mat[state]))[0][0]
    return data