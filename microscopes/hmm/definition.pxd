from microscopes._shared_ptr_h cimport shared_ptr
from microscopes.hmm._model_h cimport (
    model_definition as c_model_definition,
)

cdef class model_definition:
    cdef shared_ptr[c_model_definition] _thisptr
    cdef readonly int _N