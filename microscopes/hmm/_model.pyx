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

    def nobs(self):
      return self._thisptr.get()[0].nobs()

    def trans_mat(self):
      cdef size_t K = self.nstates()
      cdef float * pi = <float *> malloc(sizeof(float) * K * (K + 1))
      cdef float [:] pi_view = <float[:K*(K+1)]> pi
      self._thisptr.get()[0].get_pi(pi)
      trans_mat = np.ndarray(shape=(K*(K+1)), dtype=np.dtype("f"))
      cdef float [:] trans_mat_view = trans_mat
      trans_mat_view[:] = pi_view
      free(pi)
      return np.transpose(trans_mat.reshape((K+1,K)))

    def obs_mat(self):
      cdef size_t K = self.nstates()
      cdef size_t N = self.nobs()
      cdef float * phi = <float *> malloc(sizeof(float) * K * N)
      cdef float [:] phi_view = <float[:K*N]> phi
      self._thisptr.get()[0].get_phi(phi)
      obs_mat = np.ndarray(shape=(K*N), dtype=np.dtype("f"))
      cdef float [:] obs_mat_view = obs_mat
      obs_mat_view[:] = phi_view
      free(phi)
      return np.transpose(obs_mat.reshape((N,K)))

    def joint_log_likelihood(self):
      return self._thisptr.get()[0].joint_log_likelihood()