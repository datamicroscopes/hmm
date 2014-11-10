from microscopes.common import validator

cdef class model_definition:
    def __cinit__(self, N):
        validator.validate_positive(N, "N")

        self._N = N
        self._thisptr.reset(new c_model_definition(N))

    def N(self):
        return self._N
