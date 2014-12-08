from microscopes.common import validator
import numpy as np
cimport numpy as np

cdef class state:
    def __cinit__(self, model_definition defn, **kwargs):
        valid_kwargs = ('data','base','r')
        validator.validate_kwargs(kwargs, valid_kwargs)

        assert 'data' in kwargs
        data = kwargs['data']
        cdef vector[vector[size_t]] c_data = data

        assert 'r' in kwargs
        validator.validate_type(kwargs['r'], rng, param_name='r')
        cdef rng r = kwargs['r']

        # some of this should be moved to runner, but for now it's all in state
        cdef vector[float] H
        if 'base' in kwargs:
          base = kwargs['base']
        else:
          base = defn.N() * [1.0]

        self._thisptr.reset(
          new c_state(defn._thisptr.get()[0],
            base, c_data, r._thisptr[0]))

    def nstates(self):
      return self._thisptr.get()[0].nstates()

    def nobs(self):
      return self._thisptr.get()[0].nobs()

    def alpha(self):
      return self._thisptr.get()[0].alpha()

    def gamma(self):
      return self._thisptr.get()[0].gamma()

    def set_alpha_hypers(self, float alpha_a, float alpha_b):
      self._thisptr.get()[0].set_alpha_hypers(alpha_a, alpha_b)

    def set_gamma_hypers(self, float gamma_a, float gamma_b):
      self._thisptr.get()[0].set_gamma_hypers(gamma_a, gamma_b)

    def set_alpha(self, float alpha):
      self._thisptr.get()[0].set_alpha(alpha)

    def set_gamma(self, float gamma):
      self._thisptr.get()[0].set_gamma(gamma)

    def sample_hypers(self, rng r, niter=20):
      self._thisptr.get()[0].sample_hypers(r._thisptr[0],niter)

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