cdef class state:
    def __cinit__(self, model_definition defn, **kwargs):
        self._defn = defn

        self._thisptr.reset(new c_state())