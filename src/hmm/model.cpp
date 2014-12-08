#include <microscopes/hmm/model.hpp>

using namespace microscopes::hmm;

// IMPLEMENT
direct_assignment(const model_definition &defn,
                  const std::vector<float> &base,
                  distributions::rng_t &rng) {}

void direct_assignment::assign(size_t data, size_t group, size_t context) {
  MICROSCOPES_DCHECK(data < base_.size(), "Data is out of range");
  MICROSCOPES_DCHECK(group < K, "Group is out of range");
  MICROSCOPES_DCHECK(context < J, "Context is out of range");

  stick_counts_(context, group)++;
  dish_suffstats_(group, data)++;
}

void direct::assignment::remove(size_t data, size_t group, size_t context) {
  MICROSCOPES_DCHECK(data < base_.size(), "Data is out of range");
  MICROSCOPES_DCHECK(group < K, "Group is out of range");
  MICROSCOPES_DCHECK(context < J, "Context is out of range");

  MICROSCOPES_DCHECK(stick_counts_(context, group) > 0, "Cannot remove count from group");
  stick_counts_(context, group)--;

  MICROSCOPES_DCHECK(dish_suffstats_(group, data) > 0, "Cannot remove count from sufficient statistics");
  dish_suffstats_(group, data)--;
}

void direct_assignment::add_context(distributions::rng_t rng) {
  MICROSCOPES_DCHECK(sticks_.rows() == J, "Sticks Have Incorrect Number of Rows");
  MICROSCOPES_DCHECK(sticks_.cols() == K+1, "Sticks Have Incorrect Number of Cols");

  MICROSCOPES_DCHECK(stick_counts_.rows() == J, "Stick Counts Have Incorrect Number of Rows");
  MICROSCOPES_DCHECK(stick_counts_.cols() == K, "Stick Counts Have Incorrect Number of Cols");

  sticks_.conservativeResize(J+1,K+1);
  stick_counts_.conservativeResize(J+1,K);

  for (size_t i = 0; i < K; i++) {
    stick_counts_(J,i) = 0;
  }
  sample_stick_row(rng, J);
  MICROSCOPES_DCHECK(std::abs(1.0 - sticks_.block(J,0,1,K).sum()) < 1e-5, "Transition matrix row does not sum to one");
  J++;
}

void direct_assignment::add_group(distributions::rng_t rng) {
  MICROSCOPES_DCHECK(sticks_.rows() == J,   "Sticks Have Incorrect Number of Rows");
  MICROSCOPES_DCHECK(sticks_.cols() == K+1, "Sticks Have Incorrect Number of Cols");

  MICROSCOPES_DCHECK(stick_counts_.rows() == J, "Stick Counts Have Incorrect Number of Rows");
  MICROSCOPES_DCHECK(stick_counts_.cols() == K, "Stick Counts Have Incorrect Number of Cols");

  MICROSCOPES_DCHECK(dishes_.rows() == K, "Incorrect Number of Dishes");
  MICROSCOPES_DCHECK(dishes_.cols() == base_.size(), "Dishes Has Incorrect Number of Cols");

  MICROSCOPES_DCHECK(dish_suffstats_.rows() == K, "Dish Suff Stats Has Incorrect Number of Rows");
  MICROSCOPES_DCHECK(dish_suffstats_.cols() == N, "Dish Suff Stats Has Incorrect Number of Cols");

  sticks_.conservativeResize(J,K+2);
  stick_counts_.conservativeResize(J,K+1);

  for (size_t i = 0; i < J; i++) {
    stick_counts_(i,K) = 0;
  }

  dishes_.conservativeResize(K+1, base_.size());
  dish_suffstats_.conservativeResize(K+1, base_.size());

  for (size_t i = 0; i < base_.size(); i++) {
    dish_suffstats_(K,i) = 0;
  }
  sample_dish_row(rng, K);

  // Break beta stick
  float bu = beta_[K];
  float bk = distributions::sample_beta(rng, 1.0, gamma_);
  beta_[K] = bu * bk;
  beta_.push_back(bu * (1-bk));

  // Add new column to transition matrix
  max_stick = 0.0;
  for (size_t i = 0; i < J; i++) {
    float pu = sticks_(i,K);
    float pk = distributions::sample_beta(rng, alpha0_ * beta_[K], alpha0_ * beta_[K+1]);
    pi_(i,K)   = pu * pk;
    pi_(i,K+1) = pu * (1-pk);
    max_stick = max_stick > sticks_(i,K)   ? max_stick : sticks_(i,K);
    max_stick = max_stick > sticks_(i,K+1) ? max_stick : sticks_(i,K+1);
  }
  K++;
}

float direct_assignment::joint_log_likelihood() {
  float logp = 0.0;
  for (size_t k = 0; k < K; k++) {
    float count_total = alpha0_;
    for (size_t i = 0; i < K; i++) { // transition probabilities
      count_total += stick_counts_(k,i);
      if (pi_counts_(k,i) + alpha0_ * beta_[k] > 0.0) {
        logp += distributions::fast_lgamma(stick_counts_(k,i) + alpha0_ * beta_[k])
              - distributions::fast_lgamma(alpha0_ * beta_[k]);
      }
    }
    logp += distributions::fast_lgamma(alpha0_)
          - distributions::fast_lgamma(count_total);

    float H_total = 0.0;
    float H_count_total = 0.0;
    for (size_t n = 0; n < base_.size(); n++) { // emission probabilities
      H_total       += base_[n];
      H_count_total += base_[n] + dish_suffstats_(k,n);
      logp += distributions::fast_lgamma(base_[n] + dish_suffstats_(k,n))
             -distributions::fast_lgamma(base_[n]);
    }
    logp += distributions::fast_lgamma(H_total)
          - distributions::fast_lgamma(H_count_total);
  }
  return logp;
}

void state::sample_aux(distributions::rng_t &rng) {
  size_t prev_state;
  std::uniform_real_distribution<float> sampler (0.0, 1.0);
  float min_aux = 1.0; // used to figure out where to truncate sampling of pi
  for (size_t i = 0; i < data_.size(); i++) {
    for(size_t j = 0; j < data_[i].size(); j++) {
      if (j == 0) {
        prev_state = 0;
      } else {
        prev_state = states_[i][j-1];
      }
      aux_[i][j] =  sampler(rng) * (hdp_.stick(states_[i][j], prev_state)); // scale the uniform sample to be between 0 and pi_{s_{t-1}s_t}
      min_aux = min_aux < aux_[i][j] ? min_aux : aux_[i][j];
    }
  }

  // If necessary, break the pi stick some more
  while (hdp_.max_stick() > min_aux) {
    hdp_.add_context();
    hdp_.add_group();
  }
}

// FIX ME
state::state(const model_definition &defn,
      const std::vector<float> &H,
      const std::vector<std::vector<size_t> > &data,
      distributions::rng_t &rng):
  defn_(defn),
  data_(data),
  states_(data.size()),
  aux_(data.size()),
  pi_counts_(MatrixXs::Zero(1,1)),
  pi_(1,2),
  phi_counts_(MatrixXs::Zero(1,H.size())),
  phi_(1,H.size()),
  beta_(2),
  state_visited_(1),
  H_(H),
  gamma_flag_(true),
  alpha0_flag_(true),
  K(1)
{
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
    states_[i] = std::vector<size_t>(data_[i].size());
    aux_[i] = std::vector<float>(data_[i].size());
    pi_counts_(0,0) += data_[i].size();
    for (size_t j = 0; j < data_[i].size(); j++) {
      phi_counts_(0,data_[i][j])++;
    }
  }
  sample_pi(rng);
  sample_phi(rng);
}

void state::sample_state(distributions::rng_t &rng) {
  hdp_.clear(); // clear counts
  size_t K = nstates();
  state_visited_ = std::vector<bool>(K);
  for (size_t i = 0; i < data_.size(); i++) {
    // Forward-filter
    MatrixXf probs(data_[i].size(),K);
    for (size_t t = 0; t < data_[i].size(); t++) {
      float total_prob = 0.0;
      for (size_t k = 0; k < K; k++) {
        if (t == 0) {
          probs(t,k) = hdp_.dish(data_[i][t],k)
            * (aux_[i][t] <= hdp_.stick(k,0) ? 1.0 : 0.0);
        }
        else {
          probs(t,k) = 0.0;
          for (size_t l = 0; l < K; l++) {
            if (aux_[i][t] <= hdp_.stick(k,l)) {
              probs(t,k) += probs(t-1,l);
            }
          }
          probs(t,k) *= hdp_.dish(data_[i][t],k);
        }
        total_prob += probs(t,k);
      }
      MICROSCOPES_DCHECK(total_prob > 0.0, "Zero total probability");
      for (size_t k = 0; k < K; k++) { // normalize to prevent numerical underflow
        probs(t,k) /= total_prob;
      }
    }

    // Backwards-sample
    std::vector<float> likelihoods(K);
    Eigen::Map<Eigen::VectorXf> mapl(&likelihoods[0], K);
    mapl = probs.row(data_[i].size()-1);
    states_[i][data_[i].size()-1] = distributions::sample_from_likelihoods(rng, likelihoods);
    state_visited_[states_[i][data_[i].size()-1]] = true;
    for (int t = data_[i].size()-1; t > 0; t--) {
      for (size_t k = 0; k < K; k++) {
        if (aux_[i][t] >= hdp_.stick(states_[i][t],k)) {
          probs(t-1,k) = 0;
        }
      }
      mapl = probs.row(t-1);
      states_[i][t-1] = distributions::sample_from_likelihoods(rng, likelihoods);
      state_visited_[states_[i][t-1]] = true;
      hdp_.assign(data_[i][t], states_[i][t], states_[i][t-1]); // Update counts
    }
    hdp_.assign(data_[i][0], states_[i][0], 0);
  }
}

void direct_assignment::sample_hypers(distributions::rng_t &rng, bool alpha_flag, bool gamma_flag, size_t niter) {
  // sample auxiliary variable
  MatrixXs m_ = MatrixXs::Zero(K, K);
  std::uniform_real_distribution<float> sampler (0.0, 1.0);
  for (size_t i = 0; i < K; i++) {
    for (size_t j = 0; j < K; j++) {
      size_t n_ij = stick_counts_(i,j);
      if (n_ij > 0) {
        for (size_t l = 0; l < n_ij; l++) {
          if (sampler(rng) < (alpha0_ * beta_[j]) / (alpha0_ * beta_[j] + l))
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

  if (alpha_flag)
    sample_alpha(rng, m_.sum(), niter);
  if (gamma_flag)
    sample_gamma(rng, m_.sum(), niter);
}

// FIX ME
void state::clear_empty_states() {
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
          if (states_[i][t] > static_cast<size_t>(k)) states_[i][t]--;
        }
      }
      K--;
    }
  }
}

void direct_assignment::sample_gamma(distributions::rng_t &rng, size_t m, size_t iter) {
  for (size_t i = 0; i < iter; i++) {
    float mu = distributions::sample_beta(rng, gamma_ + 1, m);
    float pi_mu = 1.0 / ( 1.0 + ( m * ( hyper_gamma_b_ - distributions::fast_log(mu) ) ) / ( hyper_gamma_a_ + K - 1 ) );
    if (distributions::sample_bernoulli(rng, pi_mu)) {
      gamma_ = distributions::sample_gamma(rng, hyper_gamma_a_ + K,
        1.0 / (hyper_gamma_b_ - distributions::fast_log(mu)));
    } else {
      gamma_ = distributions::sample_gamma(rng, hyper_gamma_a_ + K + 1,
        1.0 / (hyper_gamma_b_ - distributions::fast_log(mu)));
    }
  }
}

void direct_assignment::sample_alpha(distributions::rng_t &rng, size_t m, size_t iter) {
  for (size_t i = 0; i < iter; i++) {
    float w = 0.0;
    int s = 0;
    float p;
    MatrixXs stick_counts_sum = stick_counts_.rowwise().sum();
    for (size_t k = 0; k < K; k++) {
      w += distributions::fast_log(distributions::sample_beta(rng, alpha0_ + 1, stick_counts_sum(k,0)));
      p = stick_counts_sum(k,0) / alpha0_;
      p /= p + 1;
      s += distributions::sample_bernoulli(rng, p);
      //std::cout << "w[" << k << "]:" << w << ", p[" << k << "]" << p << std::endl;
    }
    alpha0_ = distributions::sample_gamma(rng, 
      hyper_alpha_a_ + m - s,
      1.0 / (hyper_alpha_b_ - w));
  }
}

void direct_assignment::sample_sticks(distributions::rng_t &rng) {
  max_stick = 0.0;
  for (size_t k = 0; k < K; k++) {
    sample_stick_row(rng, k);
  }
}

void direct_assignment::sample_stick_row(distributions::rng_t &rng, size_t i) {
    float new_stick[K+1];
    float alphas[K+1];
    for (size_t k = 0; k < K; k++) {
      alphas[k] = stick_counts_(i,k) + alpha0_ * beta_[k];
    }
    alphas[K] = alpha0_ * beta_[K];
    distributions::sample_dirichlet(rng, K+1, alphas, new_stick);
    for (size_t k = 0; k < K+1; k++) {
      sticks_(i,k) = new_stick[k];
    }
    max_stick = max_stick > new_stick[K] ? max_stick : new_stick[K];
}

void direct_assignment::sample_dishes(distributions::rng_t &rng) {
  for (size_t k = 0; k < K; k++) {
    sample_dish_row(rng, k);
  }
}

void direct_assignment::sample_dish_row(distributions::rng_t &rng, size_t k) {
  float new_dish[defn_.N()];
  float alphas[defn_.N()];
  for (size_t n = 0; n < defn_.N(); n++) {
    alphas[n] = dish_suffstats_(k,n) + base_[n];
  }
  distributions::sample_dirichlet(rng, defn_.N(), alphas, new_dish);
  for (size_t n = 0; n < defn_.N(); n++) {
    dishes_(k,n) = new_dish[n];
  }
}