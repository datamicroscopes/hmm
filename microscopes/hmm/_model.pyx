from microscopes.common import validator
import numpy as np
cimport numpy as np

cdef class state:
    def __cinit__(self, model_definition defn, **kwargs):
        valid_kwargs = ('data','alpha','gamma','alpha_a','alpha_b','gamma_a','gamma_b','H','r')
        validator.validate_kwargs(kwargs, valid_kwargs)

        assert 'data' in kwargs
        data = kwargs['data']
        cdef vector[vector[size_t]] c_data = data

        assert 'r' in kwargs
        cdef rng r = kwargs['r']

        # some of this should be moved to runner, but for now it's all in state
        cdef vector[float] H
        if 'H' in kwargs:
          H = kwargs['H']
        else:
          H = defn.N() * [1.0]

        cdef float alpha_a, alpha_b, gamma_a, gamma_b
        cdef bool alpha_flag, gamma_flag
        if 'alpha' in kwargs:
          assert 'alpha_a' not in kwargs and 'alpha_b' not in kwargs
          alpha_a = kwargs['alpha']
          alpha_b = -1.0
          assert alpha_a > 0
        elif 'alpha_a' in kwargs and 'alpha_b' in kwargs:
          alpha_a = kwargs['alpha_a']
          alpha_b = kwargs['alpha_b']
          assert alpha_a > 0 and alpha_b > 0
        else:
          assert 'alpha_a' not in kwargs and 'alpha_b' not in kwargs
          alpha_a = 0.4
          alpha_b = -1.0

        if 'gamma' in kwargs:
          assert 'gamma_a' not in kwargs and 'gamma_b' not in kwargs
          gamma_a = kwargs['gamma']
          gamma_b = -1.0
          assert gamma_a > 0
        elif 'gamma_a' in kwargs and 'gamma_b' in kwargs:
          gamma_a = kwargs['gamma_a']
          gamma_b = kwargs['gamma_b']
          assert gamma_a > 0 and gamma_b > 0
        else:
          assert 'gamma_a' not in kwargs and 'gamma_b' not in kwargs
          gamma_a = 3.8
          gamma_b = -1.0

        self._thisptr.reset(
          new c_state(defn._thisptr.get()[0], 
            gamma_a, gamma_b, alpha_a, alpha_b, 
            H, c_data, r._thisptr[0]))

    # This will be moved to runner, but just test it here for now
    def sample(self, rng r, verbose=False):
      self._thisptr.get()[0].sample_beam(r._thisptr[0], verbose)

    def nstates(self):
      return self._thisptr.get()[0].nstates()

    def nobs(self):
      return self._thisptr.get()[0].nobs()

    def alpha(self):
      return self._thisptr.get()[0].alpha()

    def gamma(self):
      return self._thisptr.get()[0].gamma()

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