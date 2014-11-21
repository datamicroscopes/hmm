from microscopes.hmm.definition import model_definition
from microscopes.hmm.model import state
from microscopes.hmm.runner import *
from microscopes.hmm.testutil import *
from microscopes.common.rng import rng

from pprint import pprint

import numpy as np

# def test_sample_alpha():
#   data = jurgen_dataset(avglen=5000)
#   prng = rng()
#   s = state(model_definition(3), data=data, r=prng, alpha_a=4.0, alpha_b=2.0) # broad gamma hyperpriors
#   T = 5000
#   iter = 0
#   alphas = []
#   while s.joint_log_likelihood() < -4800: # burn in
#     s.sample(prng)
#     print_state(s,iter)
#     iter += 1
#   for i in xrange(T): # collect some statistics
#     s.sample(prng)
#     alphas.append(s.alpha())
#     print_state(s,iter + i)
#   print "alpha0 mean: %f, alpha0 variance: %f" % (np.mean(alphas), np.var(alphas))

# def test_sample_gamma():
#   data = jurgen_dataset(avglen=5000)
#   prng = rng()
#   s = state(model_definition(3), data=data, r=prng, gamma_a=3.0, gamma_b=6.0)
#   T = 5000
#   iter = 0
#   gammas = []
#   while s.joint_log_likelihood() < -4800: # burn in
#     s.sample(prng)
#     print_state(s,iter)
#     iter += 1
#   for i in xrange(T): # collect some statistics
#     s.sample(prng)
#     gammas.append(s.gamma())
#     print_state(s,iter+i)
#   print "gamma mean: %f, gamma variance: %f" % (np.mean(gammas), np.var(gammas))

def test_sample_hypers():
  data = jurgen_dataset(avglen=5000)
  prng = rng()
  defn = model_definition(3)
  s = state(defn, data=data, r=prng)
  r = runner(defn, data, s, default_kernel_config(defn))
  T = 5000
  iter = 0
  alphas = []
  gammas = []
  r.run(rng)
  # while s.joint_log_likelihood() < -4800: # burn in
  #   s.sample(prng)
  #   print_state(s,iter)
  #   iter += 1
  # for i in xrange(T): # collect some statistics
  #   s.sample(prng)
  #   alphas.append(s.alpha())
  #   gammas.append(s.gamma())
  #   print_state(s,iter+i)
  # print "alpha0 mean: %f, alpha0 variance: %f, gamma mean: %f, gamma variance: %f" % \
  #   (np.mean(alphas), np.var(alphas), np.mean(gammas), np.var(gammas))

# def test_beam_sampler():
#   #seed = 10758800 
#   seed = 970811
#   data = jurgen_dataset(avglen=5000, seed=seed)

#   prng = rng(seed=seed)
#   s = state(model_definition(3), data=data, r=prng)
#   lls = []
#   Ks = []
#   #T = 2218
#   T = 5618
#   for i in xrange(T):
#       s.sample(prng)
#       print "Iter: %d, States: %d, JLL: %f" % (i,s.nstates(),s.joint_log_likelihood())
#       lls.append(s.joint_log_likelihood())
#       Ks.append(s.nstates())

# def test_trans_and_obs_mat():
#   data = jurgen_dataset()
#   prng = rng()
#   s = state(model_definition(3), data=data, r=prng)
#   for i in xrange(1000):
#     s.sample(prng)
#     assert all(abs(1.0-sum(pirow))  < 1e-5 for pirow  in s.trans_mat())
#     assert all(abs(1.0-sum(phirow)) < 1e-5 for phirow in s.obs_mat())

# def test_jurgen_model():
#   """Replicate the first toy dataset from JVG 2008 """
#   data = jurgen_dataset()

#   prng = rng()
#   s = state(model_definition(3), data=data, r=prng)
#   for i in xrange(1000):
#     s.sample(prng)
#     print "%d: %d" % (i, s.nstates())

# def test_jurgen_model_2():
#   """Replicate the test dataset from JVG's MATLAB code """
#   trans_mat = np.array([[0.0, 0.5, 0.5, 0.0],
#                         [0.0, 0.0, 0.5, 0.5],
#                         [0.5, 0.0, 0.0, 0.5],
#                         [0.5, 0.5, 0.0, 0.0]])

#   obs_mat   = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
#                         [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                         [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
#                         [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]) / 3.0

#   data, _ = gen_data(trans_mat, obs_mat, avglen=500, numobs=1)
#   prng = rng()
#   s = state(model_definition(8), data=data, alpha=0.8, gamma=1, r=prng)
#   for i in xrange(1000):
#     s.sample(prng)
#     print "Iter %d, %d states, JLL=%f" % (i, s.nstates(), s.joint_log_likelihood())

# def test_beam_sampler_2():
#   defn = model_definition(10)
#   data = toy_dataset(defn)

#   prng = rng()
#   s = state(defn, data=data, r=prng)
#   s.sample(prng)

# @attr('slow')
# def test_beam_sampler_3():
#   defn = model_definition(27)
#   data = toy_dataset(defn,states=60,avglen=1000,numobs=1000)

#   prng = rng()
#   s = state(defn, data=data, r=prng)
#   for i in xrange(1000):
#     s.sample(prng)
#     print "%d: %d" % (i, s.nstates())