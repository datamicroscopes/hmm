from microscopes.common import validator

cdef class state:
    def __cinit__(self, model_definition defn, **kwargs):
        valid_kwargs = ('data',)
        validator.validate_kwargs(kwargs, valid_kwargs)

        data = kwargs['data']
        assert all(type(x) is list and all(type(y) is int for y in x) for x in data)
        cdef vector[vector[size_t]] c_data = data

        self._defn = defn

        # some of this should be moved to initialize, and some should be moved to runner, but for now it's all in state
        cdef vector[float] H = defn.N() * [1.0]
        self._thisptr.reset(new c_state(defn._thisptr.get()[0], 1.0, 1.0, H, c_data))

    # This will be moved to runner, but just test it here for now
    def sample(self, rng r):
      self._thisptr.get()[0].sample_beam(r._thisptr[0])