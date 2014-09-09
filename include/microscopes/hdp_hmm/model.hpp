#pragma once

namespace microscopes{
namespace hdp_hmm{

typedef std::vector<std::vector<size_t>> data_type; // shorten type
// To start with, just build an HDP-HMM for discrete distributions with Dirichlet prior on observations
class hdm_hmm {
public:
	hdp_hmm(size_t alphalen, data_type data) : 
		alphalen_(alphalen), 
		gamma_(),
		alpha0_(),
		obs_hp_(alphalen, 1.0 / alphalen)
		data_(data)
	{}
protected:
	// hyperparameters
	const size_t alphalen_; // alphabet size
	float gamma_; // top-level concentration parameter
	float alpha0_; // lower-level concentration parameter
	std::vector<double> obs_hp_; // vector of Dirichlet distribution hyperparameters for prior on observation distribution

	// data
	const data_type data_;

	// parameters updated by Gibbs sampling
	data_type assignments_; // assignment of every time step in the data to a latent state
	std::map<std::pair<size_t, size_t>, size_t> table_counts_; // (dish_id, restaurant_id) maps to number of tables in that restaurant that serve that dish
	std::vector<double> betas_; // stick lengths for upper level Dirichlet Process sample
};

} // namespace hdp_hmm
} // namespace microscopes