#include <microscopes/hmm/model.hpp>
#include <vector>

// kill this as soon as we integrate this into the rest of the build
int main() {
  std::vector<std::vector<size_t> > data(2);

  data[0] = std::vector<size_t>(4);
  data[0][0] = 1;
  data[0][1] = 0;
  data[0][2] = 1;
  data[0][3] = 2;

  data[1] = std::vector<size_t>(4);
  data[1][0] = 0;
  data[1][1] = 2;
  data[1][2] = 1;
  data[1][3] = 1;

  // seed random number generator
  distributions::rng_t rng;
  typedef std::chrono::high_resolution_clock myclock;
  myclock::time_point start =  myclock::now();
  myclock::duration d = myclock::now() - start;
  unsigned seed = d.count();
  rng.seed(seed);

  std::vector<float> H = {2.0,1.0,1.0};
  microscopes::hmm::hmm test(0.1, 1.0, H, data);
  for (int i = 0; i < 1000; i++) {
    std::cout << i << std::endl;
    test.sample_beam(rng);
  }

	return 0; 
}