from libcpp.vector cimport vector
from libc.stddef cimport size_t
from libcpp cimport bool

from microscopes.common._random_fwd_h cimport rng_t

cdef extern from "microscopes/hmm/model.hpp" namespace "microscopes::hmm":
  cdef cppclass model_definition:
    model_definition(size_t) except +
    size_t N()

  cdef cppclass direct_assignment:
    direct_assignment(const float,
                      const float,
                      const float,
                      const float,
                      const vector[float] &,
                      rng_t &,
                      const size_t,
                      const size_t)

    float stick(size_t, size_t)
    float dish(size_t, size_t)
    float alpha()
    float gamma()
    size_t ngroups()
    size_t ncontexts()
    float get_max_stick()

    void get_sticks(float *)
    void get_dishes(float *)

    void set_alpha_hypers(float, float)
    void set_gamma_hypers(float, float)

    void set_alpha(float)
    void set_gamma(float)

    void clear()
    void assign(size_t, size_t, size_t)
    void remove(size_t, size_t, size_t)

    void add_context(rng_t &)
    void add_group(rng_t &)
    void remove_context(size_t)
    void remove_group(size_t)

    float joint_log_likelihood()

    void sample_hypers(rng_t &, bool, bool, size_t)
    void sample_sticks(rng_t &)

  cdef cppclass state:
    state(const model_definition &,
          const float,
          const float,
          const float,
          const float,
          const vector[float] &,
          const vector[vector[size_t]],
          rng_t &)

    void get_sticks(float *)
    void get_dishes(float *)

    size_t nstates()
    size_t nobs()

    float alpha()
    float gamma()

    void set_alpha_hypers(float, float)
    void set_gamma_hypers(float, float)
    void set_alpha(float)
    void set_gamma(float)

    float joint_log_likelihood()

    void sample_aux(rng_t &)
    void sample_state(rng_t &)
    void sample_hypers(rng_t &, bool, bool, size_t)
    void clear_empty_states()