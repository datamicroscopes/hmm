#pragma once

#include <distributions/special.hpp>

#include <vector>

namespace microscopes{
namespace hmm{

	// A class for a vector of vectors, useful for representing pretty much everything we need for the beam sampler.
	// For instance, time series data can be stored as a vector of vectors, where each vector is one time series.
	// The transition matrix can also be stored as a vector of vectors, where each transition probability is one vector.
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
	class hmm {
	public:
		hmm(size_t alphalen, float gamma, float alpha0, float *H, meta_vector<size_t> data):
			gamma_(gamma),
			alpha0_(alpha0),
			H_(H),
			alphalen_(alphalen),
			data_(data),
			u_(data.size())
		{}
	protected:
		
		// parameters
		const size_t alphalen_;
		const meta_vector<size_t> data_; // XXX: For now, the observation type is just a vector of vectors of ints. Later we can switch over to using recarrays
		meta_vector<float> u_; // the slice sampling parameter for each time step in the series
		meta_vector<float> pi_; // the observed portion of the infinite transition matrix
		std::vector<float> beta_; // the stick lengths for the top-level DP draw

		// hyperparameters
		const float gamma_;
		const float alpha0_;
		const float *H_; // hyperparameters for a Dirichlet prior over observations. Will generalize this to other observation models later.

		// sampling functions. later we can integrate these into microscopes::kernels where appropriate.
		void forward_backward() {}

		void sample_u() {}

		void sample_pi() {}

		void sample_beta() {}
	};

} // namespace hmm
} // namespace microscopes