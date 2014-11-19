from microscopes.common import validator
import numpy as np
cimport numpy as np

cdef class state:
    def __cinit__(self, model_definition defn, **kwargs):
        valid_kwargs = ('data','alpha','gamma','alpha_a','alpha_b','gamma_a','gamma_b','H')
        validator.validate_kwargs(kwargs, valid_kwargs)

        data = kwargs['data']
        cdef vector[vector[size_t]] c_data = data

        # some of this should be moved to runner, but for now it's all in state
        cdef vector[float] H
        if 'H' in kwargs:
          H = kwargs['H']
        else:
          H = defn.N() * [1.0]

        cdef float alpha, gamma
        cdef bool alpha_flag, gamma_flag
        if 'alpha' in kwargs:
          assert 'alpha_a' not in kwargs and 'alpha_b' not in kwargs
          alpha_flag = False
          alpha = kwargs['alpha']
        elif 'alpha_a' in kwargs and 'alpha_b' in kwargs:
          alpha_flag = True
        else:
          assert 'alpha_a' not in kwargs and 'alpha_b' not in kwargs
          alpha_flag = False
          alpha = 0.4

        if 'gamma' in kwargs:
          assert 'gamma_a' not in kwargs and 'gamma_b' not in kwargs
          gamma_flag = False
          gamma = kwargs['gamma']
        elif 'gamma_a' in kwargs and 'gamma_b' in kwargs:
          gamma_flag = True
        else:
          assert 'gamma_a' not in kwargs and 'gamma_b' not in kwargs
          gamma_flag = False
          gamma = 3.8

        if alpha_flag and gamma_flag:
          self._thisptr.reset(
            new c_state(defn._thisptr.get()[0], gamma, alpha, H, c_data))
        elif alpha_flag and not gamma_flag:
          self._thisptr.reset(
            new c_state(defn._thisptr.get()[0], False, 
              kwargs['alpha_a'], kwargs['alpha_b'], H, c_data))
        elif not alpha_flag and gamma_flag:
          self._thisptr.reset(
            new c_state(defn._thisptr.get()[0], True,  
              kwargs['gamma_a'], kwargs['gamma_b'], H, c_data))
        else:
          self._thisptr.reset(
            new c_state(defn._thisptr.get()[0], 
              kwargs['gamma_a'], kwargs['gamma_b'], 
              kwargs['alpha_a'], kwargs['alpha_b'], H, c_data))

    # This will be moved to runner, but just test it here for now
    def sample(self, rng r, verbose=False):
      self._thisptr.get()[0].sample_beam(r._thisptr[0], verbose)

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