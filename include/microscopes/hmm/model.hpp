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

  // A class that maintains the direct assignment representation for an HDP,
  // similar to Teh et al 2006, except with explicit stick lengths for the
  // lower-level DPs as well.
  // Contains the data along with functions for hyperparameter sampling
  class direct_assignment {
  public:
    direct_assignment(const model_definition &defn,
                      const std::vector<float> &base,
                      distributions::rng_t &rng);

    inline float stick(size_t group, size_t context) { return sticks_(context, group); }
    // For general obseration models, should replace this with the score of the data under that dish
    inline float dish(size_t data, size_t group)   { return dishes_(group, data); }

    inline float alpha() { return alpha0_; }
    inline float gamma() { return gamma_; }
    inline size_t ngroups()   { return K; }
    inline size_t ncontexts() { return J; }
    inline float max_stick() { return max_stick; }

    inline void set_alpha_hypers(float hyper_alpha_a, float hyper_alpha_b) {
      hyper_alpha_a_ = hyper_alpha_a;
      hyper_alpha_b_ = hyper_alpha_b;
    }

    inline void set_gamma_hypers(float hyper_gamma_a, float hyper_gamma_b) {
      hyper_gamma_a_ = hyper_gamma_a;
      hyper_gamma_b_ = hyper_gamma_b;
    }

    inline void set_alpha(float alpha0) { alpha0_ = alpha0; }
    inline void set_gamma(float gamma)  { gamma_  = gamma; }

    inline void clear() {
      stick_counts_   = MatrixXs::Zero(J,K);
      dish_suffstats_ = MatrixXs::Zero(K,base_.size());
    }

    void assign(size_t data, size_t group, size_t context);
    void remove(size_t data, size_t group, size_t context);

    void add_context(distributions::rngt_t rng);
    void add_group(distributions::rng_t rng);

    void remove_context(size_t context);
    void remove_group(size_t group);

    float joint_log_likelihood();

    void sample_hypers(distributions::rng_t &rng, bool alpha_flag, bool gamma_flag, size_t niter);
    void sample_sticks(distributions::rng_t &rng);
    void sample_dishes(distributions::rng_t &rng);
  protected:
    std::vector<float> beta_; // the stick lengths for the top-level DP draw. Size K+1.

    // The lengths of the lower-level sticks in each context. Each row is one context.
    // There are K+1 columns and each row sums to one (the last column is all unobserved sticks)
    MatrixXf sticks_;

    // The count of how many times a sample from each stick is observed in each context.
    // Each row is one context. There are K columns.
    MatrixXs stick_counts_;

    // The dishes for each states. Once we implement more general observation models
    // this will be replaced by a length-K vector of parameter objects, but until
    // then it is a K x N matrix, where each row is the parameters for a multinomial
    MatrixXf dishes_;

    // The sufficient statistics for each dish, computed from the data assigned to each
    // stick. Once we implement more general observation models this will be a length-K
    // vector of sufficient statistic objects, but for now it's a K x N matrix where
    // each row is a vector of counts observed from a multinomial distribution
    MatrixXs dish_suffstats_;

    // hyperparameters
    float gamma_;
    float alpha0_;
    // Base distribution of the HDP. In this case,
    // hyperparameters for a Dirichlet prior over observations. 
    // Will generalize this to other observation models later.
    const std::vector<float> base_; 

    // parameters for a gamma hyperprior on gamma_ and alpha0_
    float hyper_gamma_a_, hyper_gamma_b_,
          hyper_alpha_a_, hyper_alpha_b_;

    size_t K, J; // The number of states currently instantiated, and number of contexts, respectively
    // Note, for the HDP-HMM these numbers should always be equal

    float max_stick; // the maximum value of sticks_, cached for speed
    void sample_stick_row(distributions::rng_t &rng, size_t i);
    void sample_dish_row(distributions::rng_t &rng, size_t k);
    void sample_alpha(distributions::rng_t &rng, size_t m, size_t iter);
    void sample_gamma(distributions::rng_t &rng, size_t m, size_t iter);
  };

  // Maintains the state of an HDP-HMM using the direct assignment representation for an
  // HDP, along with the auxilliary variables needed for beam sampling as in Van Gael 2008
  class state {
  public:
    state(const model_definition &defn,
          const std::vector<float> &base,
          const std::vector<std::vector<size_t> > &data,
          distributions::rng_t &rng);

    inline void get_pi(float * f)  { Eigen::Map<MatrixXf>(f, K, K+1)       = pi_; }
    inline void get_phi(float * f) { Eigen::Map<MatrixXf>(f, K, defn_.N()) = phi_; }
    inline size_t nstates()        { return hdp_.ngroups(); }
    inline size_t nobs()           { return defn_.N(); }
    inline float alpha()           { return hdp_.alpha(); }
    inline float gamma()           { return hdp_.gamma(); }

    inline void set_alpha_hypers(float a, float b) { hdp_.set_alpha_hypers(a, b); }
    inline void set_gamma_hypers(float a, float b) { hdp_.set_gamma_hypers(a, b); }
    inline void set_alpha(float alpha0) { hdp_.set_alpha(alpha0); }
    inline void set_gamma(float gamma)  { hdp_.set_gamma(gamma); }

    inline float joint_log_likelihood() { return hdp_.joint_log_likelihood();}

    void sample_aux(distributions::rng_t &rng);
    void sample_state(distributions::rng_t &rng);

    inline void sample_hypers(distributions::rng_t &rng, bool alpha_flag, bool gamma_flag, size_t niter) { 
      hdp_.sample_hypers(rng, alpha_flag, gamma_flag, niter);
    }
    void clear_empty_states();
  protected:
    const model_definition defn_;
    const std::vector<std::vector <size_t> > data_; // XXX: For now, the observation type is just a vector of vectors of ints. Later we can switch over to using recarrays
    std::vector<std::vector<size_t> > states_; // the state sequence
    std::vector<std::vector<float> > aux_; // the slice sampling parameter for each time step in the series
    direct_assignment hdp_; // the state of the HDP itself
    std::vector<bool> state_visited_; // Size K
  };

} // namespace hmm
} // namespace microscopes
