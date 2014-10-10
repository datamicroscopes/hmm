#pragma once

#include <distributions/special.hpp>
#include <distributions/random.hpp>
#include <distributions/random_fwd.hpp>

#include <vector>
#include <map>
#include <chrono>

#include <Eigen/Dense>

using namespace Eigen;
typedef Matrix<size_t, Dynamic, Dynamic> MatrixXs;

namespace microscopes{
namespace hmm{
  
  // By all rights these helper functions should be in some utility file somewhere, but I am putting them here for now
  template <typename T>
  void removeRow(T& matrix, size_t rowToRemove)
  {
      size_t numRows = matrix.rows()-1;
      size_t numCols = matrix.cols();

      if( rowToRemove < numRows )
          matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.block(rowToRemove+1,0,numRows-rowToRemove,numCols);

      matrix.conservativeResize(numRows,numCols);
  }

  template <typename T>
  void removeColumn(T& matrix, size_t colToRemove)
  {
      size_t numRows = matrix.rows();
      size_t numCols = matrix.cols()-1;

      if( colToRemove < numCols )
          matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.block(0,colToRemove+1,numRows,numCols-colToRemove);

      matrix.conservativeResize(numRows,numCols);
  }

  // A class for a vector of vectors, useful for representing pretty much everything we need for the beam sampler.
  // For instance, time series data can be stored as a vector of vectors, where each vector is one time series.
  // The transition matrix can also be stored as a vector of vectors, where each transition probability is one vector.

  template <typename T>
  class meta_vector {
  public:

    meta_vector() : data_() {}

    meta_vector(size_t size) : data_(size) {}

    meta_vector(size_t size_1, size_t size_2) : data_(size_1) {
      for (int i = 0; i < size_1; i++) {
        data_[i] = std::vector<T>(size_2);
      }
    }

    meta_vector(std::vector<size_t> size) : data_(size.size()) {
      for (int i = 0; i < data_.size(); i++) {
        data_[i] = std::vector<T>(size[i]);
      }
    }

    std::vector<T>& operator[](size_t i) {
      return data_[i];
    }

    void push_back(std::vector<T> vec) {
      data_.push_back(vec);
    }

    void erase(int i) {
      data_.erase(data_.begin()+i);
    }

    std::vector<size_t> size() {
      std::vector<size_t> sizes(0);
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
  class hmm {
  public:
    hmm(float gamma, float alpha0, const std::vector<float> &H, meta_vector<size_t> &data):
      N(H.size()),
      gamma_(gamma),
      alpha0_(alpha0),
      data_(data),
      s_(data.size()),
      u_(data.size()),
      pi_counts_(MatrixXs::Zero(1,1)),
      pi_(1,2),
      phi_counts_(MatrixXs::Zero(1,H.size())),
      phi_(1,H.size()),
      state_visited_(1),
      beta_(2),
      H_(H),
      memoized_log_stirling_(),
      K(1)
    {
      beta_[0] = 0.5;
      beta_[1] = 0.5; // simple initialization
      state_visited_[0] = true;
      for (int i = 0; i < data.size().size(); i++) {
        for (int j = 0; j < data[i].size(); j++) {
          pi_counts_(0,0)++;
          phi_counts_(0,data[i][j])++;
        }
      }
    }

    void sample_beam(distributions::rng_t &rng) {
      sample_pi(rng);
      sample_phi(rng);
      sample_u(rng);
      sample_s(rng);
      clear_empty_states();
      sample_beta(rng);
    }
  protected:
    
    // parameters

    // these three all have the same shape as the data
    meta_vector<size_t> data_; // XXX: For now, the observation type is just a vector of vectors of ints. Later we can switch over to using recarrays
    meta_vector<size_t> s_; // the state sequence
    meta_vector<float> u_; // the slice sampling parameter for each time step in the series

    // these three all have the same shape as the transition matrix, approximately
    MatrixXs pi_counts_; // the count of how many times a transition occurs between states. Size K x K.
    MatrixXf pi_; // the observed portion of the infinite transition matrix. Size K x K+1.

    MatrixXs phi_counts_; // count of how many times an observation is seen from a given state. Size K x N.
    MatrixXf phi_; // the emission matrix. Size K x N.

    // shape is the number of states currently instantiated, roughtly
    std::vector<float> beta_; // the stick lengths for the top-level DP draw. Size K+1.
    std::vector<bool> state_visited_;

    // hyperparameters
    float gamma_;
    float alpha0_;
    const std::vector<float> H_; // hyperparameters for a Dirichlet prior over observations. Will generalize this to other observation models later.

    // helper fields
    std::map<size_t, std::vector<float> > memoized_log_stirling_; // memoize computation of log stirling numbers for speed when sampling m
    // Over all instantiated states, the maximum value of the part of pi_k that belongs to the "unseen" states. 
    //Should be smaller than the least value of the auxiliary variable, so all possible states visited by the beam sampler are instantiated
    float max_pi; 
    size_t K;
    const size_t N;

    // sampling functions. later we can integrate these into microscopes::kernels where appropriate.
    void sample_s(distributions::rng_t &rng) {
      std::vector<size_t> sizes = data_.size();
      pi_counts_ = MatrixXs::Zero(K,K); // clear counts
      phi_counts_ = MatrixXs::Zero(K,N);
      state_visited_ = std::vector<bool>(K);
      for (int i = 0; i < sizes.size(); i++) {
        // Forward-filter
        MatrixXf probs(sizes[i],K);
        for (int t = 0; t < sizes[i]; t++) {
          float total_prob = 0.0;
          for (int k = 0; k < K; k++) {
            if (t == 0) {
              probs(t,k) = phi_(k,data_[i][t]) * (u_[i][t] < pi_(0,k) ? 1.0 : 0.0);
            }
            else {
              probs(t,k) = 0.0;
              for (int l = 0; l < K; l++) {
                if (u_[i][t] < pi_(l,k)) {
                  probs(t,k) += probs(t-1,l);
                }
              }
              probs(t,k) *= phi_(k,data_[i][t]);
            }
            total_prob += probs(t,k);
          }
          for (int k = 0; k < K; k++) { // normalize to prevent numerical underflow
            probs(t,k) /= total_prob;
          }
        }

        // Backwards-sample
        float * foo = probs.col(sizes[i]-1).data();
        s_[i][sizes[i]-1] = distributions::sample_from_likelihoods(rng, std::vector<float>(foo, foo + K));
        state_visited_[s_[i][sizes[i]-1]] = true;
        phi_counts_(s_[i][sizes[i]-1],data_[i][sizes[i]-1])++;
        for (int t = sizes[i]-1; t > 0; t--) {
          for (int k = 0; k < K; k++) {
            if (u_[i][t] >= pi_(k,s_[i][t])) {
              probs(t-1,k) = 0;
            }
          }
          foo = probs.col(t-1).data();
          s_[i][t-1] = distributions::sample_from_likelihoods(rng, std::vector<float>(foo, foo + K));
          // Update counts
          state_visited_[s_[i][t-1]] = true;
          pi_counts_(s_[i][t-1],s_[i][t])++;
          phi_counts_(s_[i][t-1],data_[i][t-1])++;
        }
        pi_counts_(0,s_[i][0])++; // Also add count for state 0, which is the initial state
      }
    }

    void sample_u(distributions::rng_t &rng) {
      size_t prev_state;
      std::uniform_real_distribution<float> sampler (0.0, 1.0);
      std::vector<size_t> sizes = u_.size();
      float min_u = 1.0; // used to figure out where to truncate sampling of pi
      for (int i = 0; i < sizes.size(); i++) {
        for(int j = 0; j < sizes[i]; j++) {
          if (j == 0) {
            prev_state = 0;
          } else {
            prev_state = s_[i][j-1];
          }
          u_[i][j] =  sampler(rng) / (pi_(prev_state,s_[i][j])); // scale the uniform sample to be between 0 and pi_{s_{t-1}s_t}
          min_u = min_u < u_[i][j] ? min_u : u_[i][j];
        }
      }

      // If necessary, break the pi stick some more
      while (max_pi > min_u) {
        // Add new state
        pi_.conservativeResize(K+1,K+2);
        pi_counts_.conservativeResize(K+1,K+1);

        phi_.conservativeResize(K+1,N);
        phi_counts_.conservativeResize(K+1,N);

        sample_pi_row(rng, K);
        sample_phi_row(rng, K);

        // Break beta stick
        float bu = beta_[K];
        float bk = distributions::sample_beta(rng, 1.0, gamma_);
        beta_[K] = bu * bk;
        beta_.push_back(bu * (1-bk));

        // Add new transition to each state
        max_pi = 0.0;
        for (int i = 0; i < K+1; i++) {
          float pu = pi_(i,K);
          float pk = distributions::sample_beta(rng, alpha0_ * beta_[K], alpha0_ * beta_[K+1]);
          pi_(i,K)   = pu * pk;
          pi_(i,K+1) = pu * (1-pk);
          max_pi = max_pi > pi_(i,K)   ? max_pi : pi_(i,K);
          max_pi = max_pi > pi_(i,K+1) ? max_pi : pi_(i,K+1);
        }
        K++;
      }
    }

    void clear_empty_states() {
      int cnt = 0;
      for (int k = K-1; k >= 0; k--) {
        if (!state_visited_[k]) {
          beta_[K] += beta_[k];
          beta_.erase(beta_.begin()+k);

          removeRow<MatrixXf>(phi_, k);
          removeRow<MatrixXs>(phi_counts_, k);

          removeRow<MatrixXf>(pi_, k);
          removeRow<MatrixXs>(pi_counts_, k);

          removeColumn<MatrixXf>(pi_, k);
          removeColumn<MatrixXs>(pi_counts_, k);

          // this is way inefficient and instead of relabeling states after every sample, we should probably just track which states are "active". This'll do for now.
          for (int i = 0; i < data_.size().size(); i++) {
            for (int t = 0; t < data_[i].size(); t++) {
              if (s_[i][t] > k) s_[i][t]--;
            }
          }
          K--;
        }
      }
    }

    void sample_pi(distributions::rng_t &rng) {
      max_pi = 0.0;
      for (int k = 0; k < K; k++) {
        sample_pi_row(rng, k);
      }
    }

    void sample_pi_row(distributions::rng_t &rng, size_t i) {
        float new_pi[K+1];
        float alphas[K+1];
        for (int k = 0; k < K; k++) {
          alphas[k] = pi_counts_(i,k) + alpha0_ * beta_[k];
        }
        alphas[K] = alpha0_ * beta_[K];
        distributions::sample_dirichlet(rng, K+1, alphas, new_pi);
        for (int j = 0; j < K+1; j++) {
          pi_(i,j) = new_pi[i];
        }
        max_pi = max_pi > new_pi[K] ? max_pi : new_pi[K];
    }

    void sample_phi(distributions::rng_t &rng) {
      for (int k = 0; k < K; k++) {
        sample_phi_row(rng, k);
      }
    }

    void sample_phi_row(distributions::rng_t &rng, size_t k) {
      float new_phi[N];
      float alphas[N];
      for (int n = 0; n < N; n++) {
        alphas[n] = phi_counts_(k,n) + H_[n];
      }
      distributions::sample_dirichlet(rng, N, alphas, new_phi);
      for (int n = 0; n < N; n++) {
        phi_(k,n) = new_phi[n];
      }
    }

    void sample_beta(distributions::rng_t &rng) {
      // sample auxiliary variable
      MatrixXs m_ = MatrixXs::Zero(K, K);
      for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
          size_t n_ij = pi_counts_(i,j);
          if (n_ij > 0) {
            if (!memoized_log_stirling_.count(n_ij)) {
              memoized_log_stirling_[n_ij] = distributions::log_stirling1_row(n_ij);
            }

            std::vector<float> stirling_row = memoized_log_stirling_[n_ij];

            std::vector<float> scores(n_ij);
            for (int m = 0; m < n_ij; m++) {
              scores[m] = stirling_row[m+1] + (m+1) * ( log( alpha0_ ) + log( beta_[j] ) );
            }
            m_(i,j) = distributions::sample_from_scores_overwrite(rng, scores) + 1;
          }
        }
      }

      float alphas[K+1];
      float new_beta[K+1];
      MatrixXs m_sum = m_.rowwise().sum();
      for (int k = 0; k < K; k++) {
        alphas[k] = m_sum(k);
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