#pragma once

#include <distributions/special.hpp>
#include <distributions/random.hpp>
#include <distributions/random_fwd.hpp>

#include <vector>
#include <map>

namespace microscopes{
namespace hmm{
  

  distributions::rng_t rng;

  // A class for a vector of vectors, useful for representing pretty much everything we need for the beam sampler.
  // For instance, time series data can be stored as a vector of vectors, where each vector is one time series.
  // The transition matrix can also be stored as a vector of vectors, where each transition probability is one vector.

  // Though I realize now, once I am adding-and-removing states, it might make sense to replace this with group_manager.
  template <typename T>
  class meta_vector {
  public:

    meta_vector(): data_() {}

    meta_vector(std::vector<size_t> size) : data_(size.size()) {
      for (int i = 0; i < data_.size(); i++) {
        data_[i] = std::vector<T>(size[i]);
      }
    }

    std::vector<T>& operator[](size_t i) {
      return data_[i];
    }

    std::vector<size_t> size() {
      std::vector<size_t> sizes(data_.size());
      for (std::vector<T> vec: data_) {
        sizes.push_back(vec.size());
      }
      return sizes;
    }

    T sum(size_t i) { // useful for resampling m, the number of tables serving a dish
      T result = (T)0;
      for(T t: data_[i]) {
        result += t;
      }
      return result;
    }
  protected:
    std::vector<std::vector<T> > data_;
  };

// Implementation of the beam sampler for the HDP-HMM, following van Gael 2008
  template <int N>
  class hmm {
  public:
    hmm(float gamma, float alpha0, float *H, meta_vector<size_t> data):
      gamma_(gamma),
      alpha0_(alpha0),
      H_(H),
      data_(data),
      u_(data.size()),
      beta_(),
      memoized_log_stirling_(),
      states_()
    {
    }
  protected:
    
    // parameters

    // these three all have the same shape as the data
    const meta_vector<size_t> data_; // XXX: For now, the observation type is just a vector of vectors of ints. Later we can switch over to using recarrays
    meta_vector<size_t> s_; // the state sequence
    meta_vector<float> u_; // the slice sampling parameter for each time step in the series

    // these three all have the same shape as the transition matrix
    meta_vector<size_t> m_; // auxilliary variable necessary for sampling beta
    meta_vector<size_t> counts_; // the count of how many times a transition occurs between states
    meta_vector<float> pi_; // the observed portion of the infinite transition matrix

    // shape is the number of states currently instantiated
    std::vector<float> beta_; // the stick lengths for the top-level DP draw

    // hyperparameters
    const float gamma_;
    const float alpha0_;
    const float H_[N]; // hyperparameters for a Dirichlet prior over observations. Will generalize this to other observation models later.

    // helper fields
    std::map<size_t, std::vector<float> > memoized_log_stirling_; // memoize computation of log stirling numbers for speed when sampling m
    std::set<size_t> states_;

    // sampling functions. later we can integrate these into microscopes::kernels where appropriate.
    void sample_s() {
      std::vector<size_t> sizes = data_.size();
      counts_();
      for (int i = 0; i < sizes.size(); i++) {
        // Forward-filter
        for (int t = 0; t < size[i]; t++) {

        }

        // Backwards-sample
        for (int t = size[i]-1; t >= 0; t--) {

        }
      }
    }

    void sample_u() {
      size_t prev_state;
      std::uniform_real_distribution<float> sampler (0.0, 1.0);
      std::vector<size_t> sizes = u_.size();
      for (int i = 0; i < sizes.size(); i++) {
        for(int j = 0; j < sizes[i]; j++) {
          if (j == 0) {
            prev_state = 0;
          } else {
            prev_state = s_[i][j-1];
          }
          u_[i][j] = sampler(rng) / (pi_[prev_state][s_[i][j]]); // scale the uniform sample to be between 0 and pi_{s_{t-1}s_t}
        }
      }
    }

    void sample_pi() {
      std::vector<size_t> sizes = u_.size();
      for (int i = 0; i < sizes.size(); i++) {
        float new_pi[sizes[i]+1];
        float alphas[sizes[i]+1];
        for (int j = 0; j < sizes[i]; j++) {
          alphas[j] = counts_[i][j] + alpha0_ * beta_[j];
        }
        alphas[sizes[i]] = alpha0_ * beta_[sizes[i]];
        distributions::sample_dirichlet(rng, sizes[i]+1, alphas, new_pi);
        for (int j = 0; j < sizes[i]; j++) {
          pi_[i][j] = new_pi[i];
        }
      }
    }

    void sample_m() {
      std::vector<size_t> sizes = counts_.size();
      for (int i = 0; i < sizes.size(); i++) {
        for (int j = 0; j < sizes[i]; j++) {
          size_t n_ij = counts_[i][j];
          if (!memoized_log_stirling_.count(n_ij)) {
            memoized_log_stirling_[n_ij] = distributions::log_stirling1_row(n_ij);
          }

          std::vector<float> stirling_row = memoized_log_stirling_[n_ij];

          // there's gotta be a helper function somewhere in distributions that samples from discrete log probabilities efficiently
          std::vector<float> scores(n_ij);
          for (int m = 0; m < n_ij; m++) {
            scores[m] = stirling_row[m+1] + (m+1) * ( log( alpha0_ ) + log( beta_[j] ) );
          }
          m_[i][j] = distributions::sample_from_scores_overwrite(rng, scores) + 1;
        }
      }
    }

    void sample_beta() {
      size_t K = m_.size().size();
      float alphas[K+1];
      float new_beta[K+1];
      for (int k = 0; k < K; k++) {
        alphas[k] = m_.sum(k);
      }
      alphas[K] = gamma_;
      distributions::sample_dirichlet(rng, K+1, alphas, new_beta);
      for (int k = 0; k <= K; k++) {
        beta_[k] = new_beta[k];
      }
    }
  };

} // namespace hmm
} // namespace microscopes