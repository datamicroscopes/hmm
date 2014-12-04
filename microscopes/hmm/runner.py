"""
Implements the Runner interface fo HMM
"""

from microscopes.common import validator
from microscopes.common.rng import rng
# from microscopes.common.variadic._dataview import abstract_dataview
from microscopes.hmm.definition import model_definition
from microscopes.hmm.model import state


def default_kernel_config(defn):
    """Creates a default kernel configuration suitable for general purpose
    inference.

    Parameters
    ----------
    defn : hmm definition
    """
    return [('beam', {}),
            ('hypers',
                {
                    'alpha_a': 4.0,
                    'alpha_b': 2.0,
                    'gamma_a': 3.0, 
                    'gamma_b': 6.0
                }
            )]


class runner(object):
    """The HMM runner

    Parameters
    ----------
    defn : ``model_definition``
        The structural definition.

    view : dataview
        The variadic dataview.

    latent : ``state``
        The initialization state.

    kernel_config : list. 
        Currently a dummy variable to conform to the runner interface used
        in other models, as there is only one kernel configuration implemented

    """

    def __init__(self, defn, view, latent, kernel_config):
        validator.validate_type(defn, model_definition, 'defn')
        # validator.validate_type(view, abstract_dataview, 'view') # for now, view is actually a list of lists
        validator.validate_type(latent, state, 'latent')

        self._defn = defn
        self._view = view
        self._latent = latent

        self._kernel_config = []
        for kernel in kernel_config:
            if hasattr(kernel, '__iter__'):
                name, config = kernel
            else:
                name, config = kernel, {}
            validator.validate_dict_like(config)

            if name == 'beam':
                pass
            elif name == 'hypers':
                if 'alpha' in config:
                    assert 'alpha_a' not in config and 'alpha_b' not in config
                    alpha = config['alpha']
                    assert alpha > 0
                    latent.fix_alpha(alpha)
                elif 'alpha_a' in config and 'alpha_b' in config:
                    assert 'alpha' not in config
                    alpha_a = config['alpha_a']
                    alpha_b = config['alpha_b']
                    assert alpha_a > 0 and alpha_b > 0
                    latent.set_alpha_hypers(alpha_a, alpha_b)
                else:
                    raise ValueError("Configuration missing parameters for alpha0")

                if 'gamma' in config:
                    assert 'gamma_a' not in config and 'gamma_b' not in config
                    gamma = config['gamma']
                    assert gamma > 0
                    latent.fix_gamma(gamma)
                elif 'gamma_a' in config and 'gamma_b' in config:
                    assert 'gamma' not in config
                    gamma_a = config['gamma_a']
                    gamma_b = config['gamma_b']
                    assert gamma_a > 0 and gamma_b > 0
                    latent.set_gamma_hypers(gamma_a, gamma_b)
                else:
                    raise ValueError("Configuration missing parameters for gamma")
            else:
                raise ValueError("bad kernel found: {}".format(name))

        self._kernel_config.append((name, config))

    def run(self, r, niters=10000):
        """Run the specified kernel for `niters`, in a single
        thread.

        Parameters
        ----------
        r : random state
        niters : int

        """
        validator.validate_type(r, rng, param_name='r')
        validator.validate_positive(niters, param_name='niters')
        for _ in xrange(niters):
            # This goes against every object-oriented bone in my body, but the interface must be satisfied
            # And actually Python won't even let me do this because I'm accessing a method in a C++ class...
            # I'd have to write this whole thing in Cython or change the state interface to expose all these
            # functions separately...which might actually be worth doing.
            self._latent._thisptr.get()[0].sample_aux()
            self._latent._thisptr.get()[0].sample_state()
            self._latent._thisptr.get()[0].clear_empty_states()
            self._latent._thisptr.get()[0].sample_hypers(20)
            self._latent._thisptr.get()[0].sample_pi()
            self._latent._thisptr.get()[0].sample_phi()