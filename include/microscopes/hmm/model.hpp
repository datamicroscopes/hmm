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

  // float sample_beta_robust(
  //         distributions::rng_t & rng,
  //         float alpha,
  //         float beta) {
  //     float x = distributions::sample_gamma(rng, alpha);
  //     float y = distributions::sample_gamma(rng, beta);
  //     if (x==0 && y==0) return distributions::sample_bernoulli(rng, alpha / (alpha + beta)) ? 1.0 : 0.0;
  //     return x / (x + y);
  // }

  // template<class Alloc>
  // inline size_t sample_from_likelihoods(
  //         distributions::rng_t & rng,
  //         const std::vector<float, Alloc> & likelihoods,
  //         float total_likelihood) {
  //     const size_t size = likelihoods.size();
  //     DIST_ASSERT_LT(0, size);

  //     float t = total_likelihood * distributions::sample_unif01(rng);

  //     for (size_t i = 0; DIST_LIKELY(i < size); ++i) {
  //         t -= likelihoods[i];
  //         if (DIST_UNLIKELY(t <= 0)) {
  //             return i;
  //         }
  //     }

  //     return size - 1;
  // }

  // template<class Alloc>
  // inline size_t sample_from_likelihoods(
  //         distributions::rng_t & rng,
  //         const std::vector<float, Alloc> & likelihoods) {
  //     float total = distributions::vector_sum(likelihoods.size(), likelihoods.data());
  //     return sample_from_likelihoods(rng, likelihoods, total);
  // }

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
    template <class... Args>
    static inline std::shared_ptr<state>
    initialize(Args &&... args)
    {
      return std::make_shared<state>(std::forward<Args>(args)...);
    }

    state(const model_definition &defn,
        float gamma,
        float alpha0,
        const std::vector<float> &H,
        const std::vector<std::vector<size_t> > &data,
        const distributions::rng_t &rng):
      defn_(defn),
      data_(data),
      s_(data.size()),
      u_(data.size()),
      pi_counts_(MatrixXs::Zero(1,1)),
      pi_(1,2),
      phi_counts_(MatrixXs::Zero(1,H.size())),
      phi_(1,H.size()),
      beta_(2),
      state_visited_(1),
      gamma_(gamma),
      alpha0_(alpha0),
      H_(H),
      gamma_flag_(false),
      alpha0_flag_(false),
      K(1)
    {
      init(H, rng);
    }

    state(const model_definition &defn,
        bool gamma_flag,
        float hyper_a,
        float hyper_b,
        float other_hyper,
        const std::vector<float> &H,
        const std::vector<std::vector<size_t> > &data,
        const distributions::rng_t &rng):
      defn_(defn),
      data_(data),
      s_(data.size()),
      u_(data.size()),
      pi_counts_(MatrixXs::Zero(1,1)),
      pi_(1,2),
      phi_counts_(MatrixXs::Zero(1,H.size())),
      phi_(1,H.size()),
      beta_(2),
      state_visited_(1),
      gamma_(gamma),
      alpha0_(alpha0),
      H_(H),
      gamma_flag_(gamma_flag),
      alpha0_flag_(!gamma_flag),
      K(1)
    {
      if (gamma_flag) {
        hyper_gamma_a_ = hyper_a;
        hyper_gamma_b_ = hyper_b;
        alpha0_ = other_hyper;
      } else {
        hyper_alpha_a_ = hyper_a;
        hyper_alpha_b_ = hyper_b;
        gamma_  = other_hyper;
      }
      init(H, rng);
    }

    state(const model_definition &defn,
        float hyper_gamma_a,
        float hyper_gamma_b,
        float hyper_alpha_a,
        float hyper_alpha_b,
        const std::vector<float> &H,
        const std::vector<std::vector<size_t> > &data,
        const distributions::rng_t &rng):
      defn_(defn),
      data_(data),
      s_(data.size()),
      u_(data.size()),
      pi_counts_(MatrixXs::Zero(1,1)),
      pi_(1,2),
      phi_counts_(MatrixXs::Zero(1,H.size())),
      phi_(1,H.size()),
      beta_(2),
      state_visited_(1),
      gamma_(gamma),
      alpha0_(alpha0),
      H_(H),
      gamma_flag_(true),
      alpha0_flag_(true),
      hyper_gamma_a_(hyper_gamma_a),
      hyper_gamma_b_(hyper_gamma_b),
      hyper_alpha_a_(hyper_alpha_a),
      hyper_alpha_b_(hyper_alpha_b),
      K(1)
    {
      init(H, rng);
    }

    void sample_beam(distributions::rng_t &rng, bool verbose) {
      sample_u(rng);
      sample_s(rng,verbose);
      clear_empty_states();
      sample_beta(rng);
      if (gamma_flag_)  sample_gamma(rng, 20);
      if (alpha0_flag_) sample_alpha0(rng, 20);
      sample_pi(rng);
      sample_phi(rng);
    }

    inline void get_pi(float * f)  { Eigen::Map<MatrixXf>(f, K, K+1)       = pi_; }
    inline void get_phi(float * f) { Eigen::Map<MatrixXf>(f, K, defn_.N()) = phi_; }
    inline size_t nstates()        { return K; }
    inline size_t nobs()           { return defn_.N(); }

    float joint_log_likelihood() {
      float logp = 0.0;
      for (size_t k = 0; k < K; k++) {
        float count_total = alpha0_;
        for (size_t i = 0; i < K; i++) { // transition probabilities
          count_total += pi_counts_(k,i);
          if (pi_counts_(k,i) + alpha0_ * beta_[k] > 0.0) {
            logp += distributions::fast_lgamma(pi_counts_(k,i) + alpha0_ * beta_[k])
                  - distributions::fast_lgamma(alpha0_ * beta_[k]);
          }
        }
        logp += distributions::fast_lgamma(alpha0_)
              - distributions::fast_lgamma(count_total);

        float H_total = 0.0;
        float H_count_total = 0.0;
        for (size_t n = 0; n < defn_.N(); n++) { // emission probabilities
          H_total       += H_[n];
          H_count_total += H_[n] + phi_counts_(k,n);
          logp += distributions::fast_lgamma(H_[n] + phi_counts_(k,n))
                 -distributions::fast_lgamma(H_[n]);
        }
        logp += distributions::fast_lgamma(H_total)
              - distributions::fast_lgamma(H_count_total);
      }
      return logp;
    }
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

    // Shared across the different constructors. 
    // Different calling conventions are just to distinguish different flags about sampling hyperparameters
    void init(const std::vector<float> &H, const distributions::rng_t &rng) {
      MICROSCOPES_DCHECK(H.size() == defn_.N(), "Number of hyperparameters must match vocabulary size.");
      // sample hyperparameters from prior
      if (gamma_flag_)  
        gamma_  = distributions::sample_gamma(rng, hyper_gamma_a_, 1.0 / hyper_gamma_b_); 
      if (alpha0_flag_) 
        alpha0_ = distributions::sample_gamma(rng, hyper_alpha_a_, 1.0 / hyper_alpha_b_);
      beta_[0] = 0.5;
      beta_[1] = 0.5; // simple initialization
      state_visited_[0] = true;
      for (size_t i = 0; i < data_.size(); i++) {
        s_[i] = std::vector<size_t>(data_[i].size());
        u_[i] = std::vector<float>(data_[i].size());
        pi_counts_(0,0) += data_[i].size();
        for (size_t j = 0; j < data_[i].size(); j++) {
          phi_counts_(0,data[i][j])++;
        }
      }
      sample_pi(rng);
      sample_phi(rng);
    }

    // sampling functions. later we can integrate these into microscopes::kernels where appropriate.
    void sample_s(distributions::rng_t &rng, bool verbose) {
      pi_counts_ = MatrixXs::Zero(K,K); // clear counts
      phi_counts_ = MatrixXs::Zero(K,defn_.N());
      state_visited_ = std::vector<bool>(K);
      for (size_t i = 0; i < data_.size(); i++) {
        // Forward-filter
        MatrixXf probs(data_[i].size(),K);
        for (size_t t = 0; t < data_[i].size(); t++) {
          float total_prob = 0.0;
          for (size_t k = 0; k < K; k++) {
            if (t == 0) {
              probs(t,k) = phi_(k,data_[i][t]) * (u_[i][t] <= pi_(0,k) ? 1.0 : 0.0);
            }
            else {
              probs(t,k) = 0.0;
              for (size_t l = 0; l < K; l++) {
                if (u_[i][t] <= pi_(l,k)) {
                  probs(t,k) += probs(t-1,l);
                }
              }
              probs(t,k) *= phi_(k,data_[i][t]);
            }
            total_prob += probs(t,k);
          }
          MICROSCOPES_DCHECK(total_prob > 0.0, "Zero total probability");
          for (size_t k = 0; k < K; k++) { // normalize to prevent numerical underflow
            probs(t,k) /= total_prob;
          }
          if (verbose) {
            std::cout << "Forward " << t << ": [";
            for (size_t k = 0; k < K; k++) {
              std::cout << probs(t,k);
              if (k < K-1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
          }
        }

        // Backwards-sample
        std::vector<float> likelihoods(K);
        Eigen::Map<Eigen::VectorXf> mapl(&likelihoods[0], K);
        mapl = probs.row(data_[i].size()-1);
        s_[i][data_[i].size()-1] = distributions::sample_from_likelihoods(rng, likelihoods);
        state_visited_[s_[i][data_[i].size()-1]] = true;
        phi_counts_(s_[i][data_[i].size()-1],data_[i][data_[i].size()-1])++;
        for (int t = data_[i].size()-1; t > 0; t--) {
          for (size_t k = 0; k < K; k++) {
            // if (verbose) std::cout << "u: " << u_[i][t] << ", pi: " << pi_(k,s_[i][t]) << std::endl;
            if (u_[i][t] >= pi_(k,s_[i][t])) {
              probs(t-1,k) = 0;
            }
          }
          mapl = probs.row(t-1);
          if (verbose) std::cout << "Backward " << t-1 << ": " << likelihoods << std::endl;
          s_[i][t-1] = distributions::sample_from_likelihoods(rng, likelihoods);
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
      float min_u = 1.0; // used to figure out where to truncate sampling of pi
      for (size_t i = 0; i < data_.size(); i++) {
        for(size_t j = 0; j < data_[i].size(); j++) {
          if (j == 0) {
            prev_state = 0;
          } else {
            prev_state = s_[i][j-1];
          }
          u_[i][j] =  sampler(rng) * (pi_(prev_state,s_[i][j])); // scale the uniform sample to be between 0 and pi_{s_{t-1}s_t}
          min_u = min_u < u_[i][j] ? min_u : u_[i][j];
        }
      }

      // If necessary, break the pi stick some more
      while (max_pi > min_u) {
        // Add new state
        pi_.conservativeResize(K+1,K+2);
        pi_counts_.conservativeResize(K+1,K+1);

        phi_.conservativeResize(K+1,defn_.N());
        phi_counts_.conservativeResize(K+1,defn_.N());

        for (size_t i = 0; i < K+1; i++) { // Set new counts to zero
          pi_counts_(i,K) = 0;
          pi_counts_(K,i) = 0;
        }

        for (size_t i = 0; i < defn_.N(); i++) {
          phi_counts_(K,i) = 0;
        }

        sample_pi_row(rng, K);
        MICROSCOPES_DCHECK(std::abs(1.0 - pi_.block(K,0,1,K).sum()) < 1e-5, "Transition matrix row does not sum to one");
        sample_phi_row(rng, K);

        // Break beta stick
        float bu = beta_[K];
        float bk = distributions::sample_beta(rng, 1.0, gamma_);
        beta_[K] = bu * bk;
        beta_.push_back(bu * (1-bk));

        // Add new column to transition matrix
        max_pi = 0.0;
        for (size_t i = 0; i < K+1; i++) {
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
      for (ssize_t k = K-1; k >= 0; k--) {
        if (!state_visited_[k]) {
          beta_[K] += beta_[k];
          beta_.erase(beta_.begin()+k);

          common::util::remove_row<float>(phi_, k);
          common::util::remove_row<size_t>(phi_counts_, k);

          common::util::remove_row<float>(pi_, k);
          common::util::remove_row<size_t>(pi_counts_, k);

          pi_.col(K) += pi_.col(k);
          common::util::remove_column<float>(pi_, k);
          common::util::remove_column<size_t>(pi_counts_, k);

          // this is way inefficient and instead of relabeling states after every sample
          // we should probably just track which states are "active". This'll do for now.
          for (size_t i = 0; i < data_.size(); i++) {
            for (size_t t = 0; t < data_[i].size(); t++) {
              if (s_[i][t] > static_cast<size_t>(k)) s_[i][t]--;
            }
          }
          K--;
        }
      }
    }

    void sample_pi(distributions::rng_t &rng) {
      max_pi = 0.0;
      for (size_t k = 0; k < K; k++) {
        sample_pi_row(rng, k);
      }
    }

    void sample_pi_row(distributions::rng_t &rng, size_t i) {
        float new_pi[K+1];
        float alphas[K+1];
        for (size_t k = 0; k < K; k++) {
          alphas[k] = pi_counts_(i,k) + alpha0_ * beta_[k];
        }
        alphas[K] = alpha0_ * beta_[K];
        distributions::sample_dirichlet(rng, K+1, alphas, new_pi);
        for (size_t k = 0; k < K+1; k++) {
          pi_(i,k) = new_pi[k];
        }
        max_pi = max_pi > new_pi[K] ? max_pi : new_pi[K];
    }

    void sample_phi(distributions::rng_t &rng) {
      for (size_t k = 0; k < K; k++) {
        sample_phi_row(rng, k);
      }
    }

    void sample_phi_row(distributions::rng_t &rng, size_t k) {
      float new_phi[defn_.N()];
      float alphas[defn_.N()];
      for (size_t n = 0; n < defn_.N(); n++) {
        alphas[n] = phi_counts_(k,n) + H_[n];
      }
      distributions::sample_dirichlet(rng, defn_.N(), alphas, new_phi);
      for (size_t n = 0; n < defn_.N(); n++) {
        phi_(k,n) = new_phi[n];
      }
    }

    void sample_hypers(distributions::rng_t &rng) {
      // sample auxiliary variable
      MatrixXs m_ = MatrixXs::Zero(K, K);
      std::uniform_real_distribution<float> sampler (0.0, 1.0);
      for (size_t i = 0; i < K; i++) {
        for (size_t j = 0; j < K; j++) {
          size_t n_ij = pi_counts_(i,j);
          if (n_ij > 0) {
            for (size_t l = 0; l < n_ij; l++) {
              if (sampler(rng) < (alpha0_ + beta_[j]) / (alpha0_ + beta_[j] + l))
              {
                m_(i,j)++;
              }
            }
          }
        }
      }

      float alphas[K+1];
      float new_beta[K+1];

      MatrixXs m_sum = m_.colwise().sum();
      for (size_t k = 0; k < K; k++) {
        alphas[k] = m_sum(k);
      }
      alphas[K] = gamma_;

      distributions::sample_dirichlet(rng, K+1, alphas, new_beta);
      beta_.assign(new_beta, new_beta + K+1);

      if (gamma_flag_)
        sample_gamma(rng, m_.sum(), 20);
      if (alpha0_flag_)
        sample_alpha0(rng, m_.sum(), 20);
    }

    // Resamples the hyperparameter gamma, 
    // not to be confused with distributions::sample_gamma, 
    // which samples from a Gamma distribution
    void sample_gamma(distributions::rngt_t &rng, size_t m, size_t iter) {
      for (size_t i = 0; i < iter; i++) {
        float mu = distributions::sample_beta(gamma_ + 1, m);
        float pi_mu = 1.0 / ( 1.0 + ( m * ( hyper_gamma_b_ - distributions::fast_log(mu) ) ) / ( hyper_gamma_a_ + K - 1 ) );
        if (distributions::sample_bernoulli(pi_mu)) {
          gamma = distributions::sample_gamma(hyper_gamma_a_ + K,
            1.0 / (hyper_gamma_b_ - distributions::fast_log(mu)));
        } else {
          gamma = distributions::sample_gamma(hyper_gamma_a_ + K + 1,
            1.0 / (hyper_gamma_b_ - distributions::fast_log(mu)));
        }
      }
    }

    void sample_alpha0(distributions::rng_t &rng, size_t m, size_t iter) {
      for (size_t i = 0; i < iter; i++) {
        float w = 0.0;
        int s = 0;
        float p;
        for (size_t n : pi_counts_.rowwise().sum()) {
          w += distributions::fast_log(distributions::sample_beta(rng, alpha0_ + 1, n));
          p = n / alpha0_;
          p /= p + 1;
          s += distributions::sample_bernoulli(p);
        }
        alpha0_ = distributions::sample_gamma(rng, 
          hyper_alpha_a_ + m - s,
          1.0 / (hyper_alpha_b_ - w));
      }
    }
  };

} // namespace hmm
} // namespace microscopes
