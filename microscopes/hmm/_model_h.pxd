from libcpp.vector cimport vector
from libc.stddef cimport size_t

cdef extern from "microscopes/hmm/model.hpp" namespace "microscopes::hmm":
  cdef cppclass model_definition:
    model_definition()

  cdef cppclass state:
    state()