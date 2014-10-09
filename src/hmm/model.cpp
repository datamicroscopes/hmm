#include <microscopes/hmm/model.hpp>

// kill this as soon as we integrate this into the rest of the build
int main() {
  microscopes::hmm::meta_vector<size_t> data(2,4);

  data[0][0] = 1;
  data[0][1] = 0;
  data[0][2] = 1;
  data[0][3] = 2;

  data[1][0] = 0;
  data[1][1] = 2;
  data[1][2] = 1;
  data[1][3] = 1;

  float H[3] = {2.0,1.0,1.0};
  microscopes::hmm::hmm<3> test(0.1, 1.0, H, data);
  for (int i = 0; i < 1000; i++) {
    std::cout << i << std::endl;
    test.sample_beam();
  }

	return 0; 
}