cdef class state:
    def __cinit__(self, model_definition defn, **kwargs):
        valid_kwargs = ('data',)
        validator.validate_kwargs(kwargs, valid_kwargs)

        data = kwargs['data']
        # validator.validate_type(data, vector[vector[size_t]], 'data')

        self._defn = defn

        cdef vector[float] H = defn.N() * [1.0]
        self._thisptr.reset(new c_state(defn._thisptr.get()[0], 1.0, 1.0, H, data))