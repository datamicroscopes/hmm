#pragma once

#include <distributions/special.hpp>
#include <distributions/random.hpp>
#include <distributions/random_fwd.hpp>

#include <microscopes/common/util.hpp>

#include <vector>
#include <map>
#include <chrono>
#include <cmath>

#include <eigen3/Eigen/Dense>

namespace microscopes{
namespace hmm{

  typedef Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;
  typedef Eigen::MatrixXf MatrixXf;

  class model_definition {
  public:
    /**
    * N - size of observation vocabulary
    **/
    model_definition(size_t N): N_(N)
    {
      MICROSCOPES_DCHECK(N > 0, "Vocabulary cannot be empty");
    }
    inline size_t N() const { return N_; }
  protected:
    size_t N_;
  };

// Implementation of the beam sampler for the HDP-HMM, following van Gael 2008
  class state {
  public:
    state(const model_definition &defn,
          const std::vector<float> &H,
          const std::vector<std::vector<size_t> > &data,
          distributions::rng_t &rng);

    inline void get_pi(float * f)  { Eigen::Map<MatrixXf>(f, K, K+1)       = pi_; }
    inline void get_phi(float * f) { Eigen::Map<MatrixXf>(f, K, defn_.N()) = phi_; }
    inline size_t nstates()        { return K; }
    inline size_t nobs()           { return defn_.N(); }
    inline float alpha()           { return alpha0_; }
    inline float gamma()           { return gamma_; }

    inline void set_alpha_hypers(float hyper_alpha_a, float hyper_alpha_b) {
      alpha0_flag_ = true;
      hyper_alpha_a_ = hyper_alpha_a;
      hyper_alpha_b_ = hyper_alpha_b;
    }

    inline void set_gamma_hypers(float hyper_gamma_a, float hyper_gamma_b) {
      gamma_flag_ = true;
      hyper_gamma_a_ = hyper_gamma_a;
      hyper_gamma_b_ = hyper_gamma_b;
    }

    inline void fix_alpha(float alpha0) {
      alpha0_flag_ = false;
      alpha0_ = alpha0;
    }

    inline void fix_gamma(float gamma) {
      gamma_flag_ = false;
      gamma_ = gamma;
    }

    float joint_log_likelihood();

    void sample_aux(distributions::rng_t &rng);
    void sample_state(distributions::rng_t &rng);
    void sample_hypers(distributions::rng_t &rng, size_t niter);

    void clear_empty_states();

    void sample_pi(distributions::rng_t &rng);
    void sample_phi(distributions::rng_t &rng);

  protected:

    // parameters

    // these three all have the same shape as the data
    const model_definition defn_;
    const std::vector<std::vector <size_t> > data_; // XXX: For now, the observation type is just a vector of vectors of ints. Later we can switch over to using recarrays
    std::vector<std::vector<size_t> > s_; // the state sequence
    std::vector<std::vector<float> > u_; // the slice sampling parameter for each time step in the series

    // same shape as the transition matrix, or plus one column
    MatrixXs pi_counts_; // the count of how many times a transition occurs between states. Size K x K.
    MatrixXf pi_; // the observed portion of the infinite transition matrix. Size K x K+1.

    // same shape as the observation matrix
    MatrixXs phi_counts_; // count of how many times an observation is seen from a given state. Size K x N.
    MatrixXf phi_; // the emission matrix. Size K x N.

    std::vector<float> beta_; // the stick lengths for the top-level DP draw. Size K+1.
    std::vector<bool> state_visited_; // Size K

    // hyperparameters
    float gamma_;
    float alpha0_;
    const std::vector<float> H_; // hyperparameters for a Dirichlet prior over observations. Will generalize this to other observation models later.

    // If true, resample the hyperparameter in each loop
    bool gamma_flag_, alpha0_flag_; 
    // Only assigned values if the corresponding flag is true
    float hyper_gamma_a_, hyper_gamma_b_,
          hyper_alpha_a_, hyper_alpha_b_;

    // helper fields
    // Over all instantiated states, the maximum value of the part of pi_k that belongs to the "unseen" states.
    //Should be smaller than the least value of the auxiliary variable, so all possible states visited by the beam sampler are instantiated
    float max_pi;
    size_t K;

    void sample_pi_row(distributions::rng_t &rng, size_t i);
    void sample_phi_row(distributions::rng_t &rng, size_t k);

    // Resamples the hyperparameter gamma, 
    // not to be confused with distributions::sample_gamma, 
    // which samples from a Gamma distribution
    void sample_gamma(distributions::rng_t &rng, size_t m, size_t iter);
    void sample_alpha0(distributions::rng_t &rng, size_t m, size_t iter);
  };

} // namespace hmm
} // namespace microscopes
