from microscopes._shared_ptr_h cimport shared_ptr
from microscopes.common._random_fwd_h cimport rng_t
from microscopes.hmm._model_h cimport state as c_state
from microscopes.hmm.definition cimport model_definition
from microscopes.common._rng cimport rng

from libcpp.vector cimport vector
from libcpp.stddef import size_t
from libc.stdlib cimport malloc, free

cdef class state:
    cdef shared_ptr[c_state] _thisptr