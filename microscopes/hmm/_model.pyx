from microscopes.common import validator
import numpy as np
cimport numpy as np

cdef class state:
    def __cinit__(self, model_definition defn, alpha=0.4, gamma=3.8, **kwargs):
        valid_kwargs = ('data',)
        validator.validate_kwargs(kwargs, valid_kwargs)

        data = kwargs['data']
        # assert all(type(x) is list and all(type(y) is int for y in x) for x in data)
        cdef vector[vector[size_t]] c_data = data

        # some of this should be moved to initialize, and some should be moved to runner, but for now it's all in state
        cdef vector[float] H = defn.N() * [1.0]
        self._thisptr.reset(new c_state(defn._thisptr.get()[0], gamma, alpha, H, c_data))

    # This will be moved to runner, but just test it here for now
    def sample(self, rng r):
      self._thisptr.get()[0].sample_beam(r._thisptr[0])

    def nstates(self):
      return self._thisptr.get()[0].nstates()

    def trans_mat(self):
      pass

    def obs_mat(self):
      pass

    def joint_log_likelihood(self):
      return self._thisptr.get()[0].joint_log_likelihood();