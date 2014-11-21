from libcpp.vector cimport vector
from libc.stddef cimport size_t
from libcpp cimport bool

from microscopes.common._random_fwd_h cimport rng_t

cdef extern from "microscopes/hmm/model.hpp" namespace "microscopes::hmm":
  cdef cppclass model_definition:
    model_definition(size_t) except +
    size_t N()

  cdef cppclass state:
    state(const model_definition &,
          const vector[float] &,
          const vector[vector[size_t]],
          rng_t &)

    void sample_beam(rng_t &, bool)

    size_t nstates()
    size_t nobs()

    void get_pi(float *)
    void get_phi(float *)

    float joint_log_likelihood()

    float alpha()
    float gamma()

    void set_alpha_hypers(float, float)
    void set_gamma_hypers(float, float)
    void fix_alpha(float)
    void fix_gamma(float)

    void sample_aux(rng_t &)
    void sample_state(rng_t &)
    void sample_hypers(rng_t &, size_t)
    void sample_pi(rng_t &)
    void sample_phi(rng_t &)