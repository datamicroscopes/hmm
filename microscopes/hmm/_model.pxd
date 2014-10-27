from microscopes._shared_ptr_h cimport shared_ptr
from microscopes.common._random_fwd_h cimport rng_t
from microscopes.hmm._model_h cimport (
    model_definition as c_model_definition,
    state as c_state,
)

cdef class state:
    cdef shared_ptr[c_state] _thisptr
    cdef readonly void sample_beam(rng_t &)