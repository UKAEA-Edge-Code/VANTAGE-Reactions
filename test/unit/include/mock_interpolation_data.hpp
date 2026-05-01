#ifndef REACTIONS_MOCK_INTERPOLATION_DATA_H
#define REACTIONS_MOCK_INTERPOLATION_DATA_H
#include <memory>
#include <neso_particles.hpp>
#include <neso_particles/typedefs.hpp>
#include <reactions/reactions.hpp>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

#include "reactions_lib/reaction_data/grid_eval_data.hpp"
#include "reactions_lib/reaction_data/trim_eval_data.hpp"

namespace test_composite_data {
// Using-declarations for backward compatibility with test code
template <int N>
using GridEvalOnDevice = VANTAGE::Reactions::GridEvalOnDevice<N>;

template <int N> using GridEval = VANTAGE::Reactions::GridEval<N>;

template <int I, int O>
using TrimEvalOnDevice = VANTAGE::Reactions::TrimEvalOnDevice<I, O>;

template <int I, int O> using TrimEval = VANTAGE::Reactions::TrimEval<I, O>;
} // namespace test_composite_data

struct abstract_coefficient_values {
  abstract_coefficient_values() = default;

  virtual ~abstract_coefficient_values() = default;

protected:
  std::vector<REAL> coeffs_vec;
  std::vector<REAL> ranges_flat_vec;
  std::vector<REAL> lower_bounds;
  std::vector<REAL> upper_bounds;
  std::vector<size_t> dims_vec;
  std::optional<SYCLTargetSharedPtr> sycl_target;

public:
  const std::vector<REAL> &get_coeffs_vec() { return this->coeffs_vec; }

  const std::vector<REAL> &get_ranges_flat_vec() {
    return this->ranges_flat_vec;
  }

  const std::vector<size_t> &get_dims_vec() { return this->dims_vec; }

  const std::vector<REAL> &get_lower_bounds() { return this->lower_bounds; }

  const std::vector<REAL> &get_upper_bounds() { return this->upper_bounds; }

  template <int input_ndim> auto get_grid_func_data_ndim() const {
    std::optional<test_composite_data::GridEval<input_ndim>> return_val;
    if (this->sycl_target) {
      return_val = test_composite_data::GridEval<input_ndim>(
          this->coeffs_vec, this->ranges_flat_vec, this->dims_vec,
          this->sycl_target.value());
    }
    // deliberately error if no value is set.
    return return_val.value();
  }
};

struct coefficient_values_1D : abstract_coefficient_values {
private:
  static constexpr int ndim = 1;
  const int dim0 = 8;

  // Generated with python: numpy.linspace(1.0e18, 8.0e18, 8)
  std::vector<REAL> dim0_range = {1.0e+18, 2.0e+18, 3.0e+18, 4.0e+18,
                                  5.0e+18, 6.0e+18, 7.0e+18, 8.0e+18};

  static constexpr auto grid_func_lambda = [](const REAL &dim0_val) {
    return (2 * dim0_val);
  };

  utils::LambdaWrapper<decltype(grid_func_lambda)> grid_func;

public:
  coefficient_values_1D(
      std::optional<SYCLTargetSharedPtr> sycl_target = std::nullopt)
      : abstract_coefficient_values(), grid_func(grid_func_lambda) {

    this->sycl_target = sycl_target;

    REAL dim0_i = 0.0;
    REAL val = 0.0;
    for (int idim0 = 0; idim0 < this->dim0; idim0++) {
      dim0_i = this->dim0_range[idim0];
      val = this->grid_func(dim0_i);
      this->coeffs_vec.push_back(val);
    }

    this->ranges_flat_vec = this->dim0_range;

    this->lower_bounds.push_back(this->dim0_range[0]);
    this->upper_bounds.push_back(this->dim0_range[this->dim0 - 1]);

    this->dims_vec.push_back(this->dim0);
  };

  auto get_grid_func() const { return this->grid_func; }

  auto get_grid_func_data() const {
    return this->get_grid_func_data_ndim<ndim>();
  }
};

struct coefficient_values_2D : abstract_coefficient_values {
private:
  static constexpr int ndim = 2;
  const size_t dim0 = 8;
  const size_t dim1 = 10;

  // Generated with python: numpy.linspace(1.0e18, 8.0e18, 8)
  std::vector<REAL> dim0_range = {1.0e+18, 2.0e+18, 3.0e+18, 4.0e+18,
                                  5.0e+18, 6.0e+18, 7.0e+18, 8.0e+18};
  // Generated with python: numpy.logspace(1, 5, 10)
  std::vector<REAL> dim1_range = {
      1.00000000e+01, 2.78255940e+01, 7.74263683e+01, 2.15443469e+02,
      5.99484250e+02, 1.66810054e+03, 4.64158883e+03, 1.29154967e+04,
      3.59381366e+04, 1.00000000e+05};

  static constexpr auto grid_func_lambda = [](const REAL &dim0_val,
                                              const REAL &dim1_val) {
    return (dim0_val * dim1_val);
  };

  utils::LambdaWrapper<decltype(grid_func_lambda)> grid_func;

public:
  coefficient_values_2D(
      std::optional<SYCLTargetSharedPtr> sycl_target = std::nullopt)
      : abstract_coefficient_values(), grid_func(grid_func_lambda) {

    this->sycl_target = sycl_target;

    REAL dim1_i = 0.0;
    REAL dim0_i = 0.0;
    REAL val = 0.0;
    for (int idim1 = 0; idim1 < this->dim1; idim1++) {
      dim1_i = this->dim1_range[idim1];
      for (int idim0 = 0; idim0 < this->dim0; idim0++) {
        dim0_i = this->dim0_range[idim0];
        val = this->grid_func(dim0_i, dim1_i);
        this->coeffs_vec.push_back(val);
      }
    }

    this->ranges_flat_vec = this->dim0_range;
    this->ranges_flat_vec.insert(this->ranges_flat_vec.end(),
                                 this->dim1_range.begin(),
                                 this->dim1_range.end());

    this->lower_bounds.push_back(this->dim0_range[0]);
    this->lower_bounds.push_back(this->dim1_range[0]);

    this->upper_bounds.push_back(this->dim0_range[this->dim0 - 1]);
    this->upper_bounds.push_back(this->dim1_range[this->dim1 - 1]);

    this->dims_vec.push_back(this->dim0);
    this->dims_vec.push_back(this->dim1);
  };

  auto get_grid_func() const { return this->grid_func; }

  auto get_grid_func_data() const {
    return this->get_grid_func_data_ndim<ndim>();
  }
};

struct trim_coefficient_values : abstract_coefficient_values {
private:
  static constexpr int ndim = 2;
  static constexpr int trim_ndim = 3;

  static constexpr int dim0 = 100;
  static constexpr int dim1 = 70;

  static constexpr int trim_dim0 = 5;
  static constexpr int trim_dim1 = 5;
  static constexpr int trim_dim2 = 5;

  static constexpr REAL c1dtd1 = 1.0 / static_cast<REAL>(trim_dim1);
  static constexpr REAL c1dtd2 = 1.0 / static_cast<REAL>(trim_dim2);
  static constexpr REAL c1dtd1td2 =
      1.0 / static_cast<REAL>(trim_dim1 * trim_dim2);

  static inline std::vector<size_t> trim_dims_vec{trim_dim0, trim_dim1,
                                                  trim_dim2};

  // generated using: numpy.logspace(0, numpy.log10(5e3), 100)
  const std::vector<REAL> dim0_range = {
      1.00000000e+00, 1.08984148e+00, 1.18775445e+00, 1.29446407e+00,
      1.41076064e+00, 1.53750546e+00, 1.67563723e+00, 1.82617896e+00,
      1.99024558e+00, 2.16905218e+00, 2.36392304e+00, 2.57630139e+00,
      2.80776012e+00, 3.06001344e+00, 3.33492958e+00, 3.63454458e+00,
      3.96107745e+00, 4.31694651e+00, 4.70478737e+00, 5.12747243e+00,
      5.58813215e+00, 6.09017821e+00, 6.63732883e+00, 7.23363628e+00,
      7.88351687e+00, 8.59178369e+00, 9.36368225e+00, 1.02049293e+01,
      1.11217553e+01, 1.21209502e+01, 1.32099143e+01, 1.43967126e+01,
      1.56901346e+01, 1.70997595e+01, 1.86360272e+01, 2.03103154e+01,
      2.21350242e+01, 2.41236676e+01, 2.62909736e+01, 2.86529935e+01,
      3.12272209e+01, 3.40327206e+01, 3.70902706e+01, 4.04225154e+01,
      4.40541340e+01, 4.80120226e+01, 5.23254938e+01, 5.70264936e+01,
      6.21498382e+01, 6.77334716e+01, 7.38187469e+01, 8.04507324e+01,
      8.76785453e+01, 9.55557156e+01, 1.04140582e+02, 1.13496727e+02,
      1.23693440e+02, 1.34806242e+02, 1.46917434e+02, 1.60116714e+02,
      1.74501837e+02, 1.90179340e+02, 2.07265333e+02, 2.25886358e+02,
      2.46180322e+02, 2.68297527e+02, 2.92401774e+02, 3.18671582e+02,
      3.47301508e+02, 3.78503590e+02, 4.12508913e+02, 4.49569324e+02,
      4.89959297e+02, 5.33977966e+02, 5.81951336e+02, 6.34234706e+02,
      6.91215290e+02, 7.53315095e+02, 8.20994038e+02, 8.94753358e+02,
      9.75139324e+02, 1.06274728e+03, 1.15822607e+03, 1.26228282e+03,
      1.37568817e+03, 1.49928203e+03, 1.63397975e+03, 1.78077891e+03,
      1.94076672e+03, 2.11512808e+03, 2.30515432e+03, 2.51225279e+03,
      2.73795730e+03, 2.98393944e+03, 3.25202097e+03, 3.54418735e+03,
      3.86260238e+03, 4.20962430e+03, 4.58782318e+03, 5.00000000e+03};

  // generated using: 90.0 - numpy.logspace(numpy.log10(90.0),
  // numpy.log10(90.0 - 8.5e1), 70) just replace the first element to 0.0;
  const std::vector<REAL> dim1_range = {
      0.00000000e+00, 3.69217858e+00, 7.23288847e+00, 1.06283435e+01,
      1.38845028e+01, 1.70070806e+01, 2.00015572e+01, 2.28731878e+01,
      2.56270120e+01, 2.82678627e+01, 3.08003747e+01, 3.32289923e+01,
      3.55579779e+01, 3.77914186e+01, 3.99332343e+01, 4.19871836e+01,
      4.39568713e+01, 4.58457541e+01, 4.76571470e+01, 4.93942289e+01,
      5.10600485e+01, 5.26575291e+01, 5.41894743e+01, 5.56585727e+01,
      5.70674025e+01, 5.84184362e+01, 5.97140448e+01, 6.09565021e+01,
      6.21479885e+01, 6.32905952e+01, 6.43863273e+01, 6.54371079e+01,
      6.64447811e+01, 6.74111153e+01, 6.83378063e+01, 6.92264806e+01,
      7.00786978e+01, 7.08959534e+01, 7.16796817e+01, 7.24312583e+01,
      7.31520019e+01, 7.38431777e+01, 7.45059985e+01, 7.51416276e+01,
      7.57511806e+01, 7.63357271e+01, 7.68962930e+01, 7.74338622e+01,
      7.79493780e+01, 7.84437452e+01, 7.89178314e+01, 7.93724686e+01,
      7.98084546e+01, 8.02265547e+01, 8.06275025e+01, 8.10120018e+01,
      8.13807273e+01, 8.17343261e+01, 8.20734188e+01, 8.23986005e+01,
      8.27104419e+01, 8.30094903e+01, 8.32962704e+01, 8.35712856e+01,
      8.38350185e+01, 8.40879319e+01, 8.43304698e+01, 8.45630578e+01,
      8.47861041e+01, 8.50000000e+01};

  static const inline auto trim_grid_func_0 =
      [](const REAL &dim0_val, const REAL &dim1_val,
         const std::array<REAL, trim_ndim> &rand_nums) {
        std::array<REAL, trim_dim0> result;
        for (int idim = 0; idim < trim_dim0; idim++) {
          result[idim] = (dim0_val * dim1_val);
          result[idim] *=
              rand_nums[0] * Kernel::pow(static_cast<REAL>(idim), 4.0);
        }
        return result;
      };

  static const inline auto trim_grid_func_1 =
      [](const REAL &dim0_val, const REAL &dim1_val,
         const std::array<REAL, trim_ndim> &rand_nums) {
        std::array<REAL, trim_dim0 * trim_dim1> result;
        for (INT counter = 0; counter < trim_dim0 * trim_dim1; counter++) {
          INT idim = static_cast<INT>(static_cast<REAL>(counter) * c1dtd1);
          INT jdim = counter % trim_dim1;
          result[counter] = counter * (dim0_val * dim1_val);
          result[counter] *=
              (rand_nums[0] * Kernel::pow(static_cast<REAL>(idim), 4.0) +
               rand_nums[1] * Kernel::pow(static_cast<REAL>(jdim), 3.0));
        }
        return result;
      };

  static const inline auto trim_grid_func_2 =
      [](const REAL &dim0_val, const REAL &dim1_val,
         const std::array<REAL, trim_ndim> &rand_nums) {
        std::array<REAL, trim_dim0 * trim_dim1 * trim_dim2> result;
        for (INT counter = 0; counter < trim_dim0 * trim_dim1 * trim_dim2;
             counter++) {
          INT idim = static_cast<INT>(static_cast<REAL>(counter) * c1dtd1td2);
          INT jdim =
              static_cast<INT>(static_cast<REAL>(counter) * c1dtd2) % trim_dim1;
          INT kdim = counter % trim_dim2;
          result[counter] = counter * (dim0_val * dim1_val);
          result[counter] *=
              (rand_nums[0] * Kernel::pow(static_cast<REAL>(idim), 4.0) +
               rand_nums[1] * Kernel::pow(static_cast<REAL>(jdim), 3.0) +
               rand_nums[2] * Kernel::pow(static_cast<REAL>(kdim), 2.0));
        }
        return result;
      };

  static const inline auto trim_grid_func_lambda =
      [](const REAL &dim0_val, const REAL &dim1_val,
         const std::array<INT, trim_ndim> &trim_indices,
         const std::array<INT, trim_ndim> &trim_dims,
         const std::array<REAL, trim_ndim> &rand_nums) {
        auto trim_vals_trim_dim0 =
            trim_grid_func_0(dim0_val, dim1_val, rand_nums)[trim_indices[0]];

        auto trim_vals_trim_dim1 = trim_grid_func_1(
            dim0_val, dim1_val,
            rand_nums)[(trim_indices[0] * trim_dims[1]) + trim_indices[1]];

        auto trim_vals_trim_dim2 = trim_grid_func_2(
            dim0_val, dim1_val,
            rand_nums)[(trim_indices[0] * (trim_dims[2] * trim_dims[1])) +
                       (trim_indices[1] * trim_dims[2]) + trim_indices[2]];

        return std::array<REAL, trim_ndim>{
            trim_vals_trim_dim0, trim_vals_trim_dim1, trim_vals_trim_dim2};
      };

  utils::LambdaWrapper<decltype(trim_grid_func_lambda)> trim_grid_func;

public:
  trim_coefficient_values(
      const std::array<REAL, trim_ndim> &rand_nums,
      std::optional<SYCLTargetSharedPtr> sycl_target = std::nullopt)
      : abstract_coefficient_values(), trim_grid_func(trim_grid_func_lambda) {

    this->sycl_target = sycl_target;

    std::vector<std::vector<std::array<REAL, trim_dim0>>> trim_vals_trim_dim0(
        dim1, std::vector<std::array<REAL, trim_dim0>>(dim0));
    std::vector<std::vector<std::array<REAL, trim_dim0 * trim_dim1>>>
        trim_vals_trim_dim1(
            dim1, std::vector<std::array<REAL, trim_dim0 * trim_dim1>>(dim0));
    std::vector<
        std::vector<std::array<REAL, trim_dim0 * trim_dim1 * trim_dim2>>>
        trim_vals_trim_dim2(
            dim1,
            std::vector<std::array<REAL, trim_dim0 * trim_dim1 * trim_dim2>>(
                dim0));

    std::vector<std::vector<std::vector<REAL>>> trim_vals(
        dim1, std::vector<std::vector<REAL>>(dim0));

    for (int idim1 = 0; idim1 < this->dim1; idim1++) {
      for (int idim0 = 0; idim0 < this->dim0; idim0++) {
        trim_vals_trim_dim0[idim1][idim0] = trim_grid_func_0(
            this->dim0_range[idim0], this->dim1_range[idim1], rand_nums);

        trim_vals_trim_dim1[idim1][idim0] = trim_grid_func_1(
            this->dim0_range[idim0], this->dim1_range[idim1], rand_nums);

        trim_vals_trim_dim2[idim1][idim0] = trim_grid_func_2(
            this->dim0_range[idim0], this->dim1_range[idim1], rand_nums);

        for (int itrim_dim0 = 0; itrim_dim0 < trim_dim0; itrim_dim0++) {
          trim_vals[idim1][idim0].push_back(
              trim_vals_trim_dim0[idim1][idim0][itrim_dim0]);
        }

        for (int itrim_dim1 = 0; itrim_dim1 < (trim_dim0 * trim_dim1);
             itrim_dim1++) {
          trim_vals[idim1][idim0].push_back(
              trim_vals_trim_dim1[idim1][idim0][itrim_dim1]);
        }

        for (int itrim_dim2 = 0;
             itrim_dim2 < (trim_dim0 * trim_dim1 * trim_dim2); itrim_dim2++) {
          trim_vals[idim1][idim0].push_back(
              trim_vals_trim_dim2[idim1][idim0][itrim_dim2]);
        }

        this->coeffs_vec.insert(this->coeffs_vec.end(),
                                trim_vals[idim1][idim0].begin(),
                                trim_vals[idim1][idim0].end());
      }
    }

    this->ranges_flat_vec = this->dim0_range;
    this->ranges_flat_vec.insert(this->ranges_flat_vec.end(),
                                 this->dim1_range.begin(),
                                 this->dim1_range.end());

    this->lower_bounds.push_back(this->dim0_range[0]);
    this->lower_bounds.push_back(this->dim1_range[0]);

    this->upper_bounds.push_back(this->dim0_range[this->dim0 - 1]);
    this->upper_bounds.push_back(this->dim1_range[this->dim1 - 1]);

    this->dims_vec.push_back(this->dim0);
    this->dims_vec.push_back(this->dim1);
  }

  auto get_trim_dims_vec() const { return this->trim_dims_vec; }

  auto get_grid_func_data() const {
    std::optional<test_composite_data::TrimEval<ndim + trim_ndim, trim_ndim>>
        return_val;
    if (this->sycl_target) {
      return_val = test_composite_data::TrimEval<ndim + trim_ndim, trim_ndim>(
          this->coeffs_vec, this->ranges_flat_vec, this->dims_vec,
          this->trim_dims_vec, this->sycl_target.value());
    }
    return return_val.value();
  }

  auto get_grid_func() const { return this->trim_grid_func; }
};

struct trim_coefficient_values_asym : abstract_coefficient_values {
private:
  static constexpr int ndim = 2;
  static constexpr int trim_ndim = 3;

  static constexpr int dim0 = 100;
  static constexpr int dim1 = 70;

  static constexpr int trim_dim0 = 3;
  static constexpr int trim_dim1 = 5;
  static constexpr int trim_dim2 = 4;

  static constexpr REAL c1dtd1 = 1.0 / static_cast<REAL>(trim_dim1);
  static constexpr REAL c1dtd2 = 1.0 / static_cast<REAL>(trim_dim2);
  static constexpr REAL c1dtd1td2 =
      1.0 / static_cast<REAL>(trim_dim1 * trim_dim2);

  static inline std::vector<size_t> trim_dims_vec{trim_dim0, trim_dim1,
                                                  trim_dim2};

  // generated using: numpy.logspace(1e4, numpy.log10(4e4), 100)
  const std::vector<REAL> dim0_range = {
      1.0000e+04, 1.0141e+04, 1.0284e+04, 1.0429e+04, 1.0576e+04, 1.0725e+04,
      1.0876e+04, 1.1030e+04, 1.1185e+04, 1.1343e+04, 1.1503e+04, 1.1665e+04,
      1.1830e+04, 1.1997e+04, 1.2166e+04, 1.2337e+04, 1.2511e+04, 1.2688e+04,
      1.2867e+04, 1.3048e+04, 1.3232e+04, 1.3419e+04, 1.3608e+04, 1.3800e+04,
      1.3994e+04, 1.4192e+04, 1.4392e+04, 1.4595e+04, 1.4801e+04, 1.5009e+04,
      1.5221e+04, 1.5436e+04, 1.5653e+04, 1.5874e+04, 1.6098e+04, 1.6325e+04,
      1.6555e+04, 1.6789e+04, 1.7025e+04, 1.7265e+04, 1.7509e+04, 1.7756e+04,
      1.8006e+04, 1.8260e+04, 1.8517e+04, 1.8779e+04, 1.9043e+04, 1.9312e+04,
      1.9584e+04, 1.9860e+04, 2.0141e+04, 2.0425e+04, 2.0713e+04, 2.1005e+04,
      2.1301e+04, 2.1601e+04, 2.1906e+04, 2.2215e+04, 2.2528e+04, 2.2846e+04,
      2.3168e+04, 2.3495e+04, 2.3826e+04, 2.4162e+04, 2.4503e+04, 2.4848e+04,
      2.5198e+04, 2.5554e+04, 2.5914e+04, 2.6280e+04, 2.6650e+04, 2.7026e+04,
      2.7407e+04, 2.7793e+04, 2.8185e+04, 2.8583e+04, 2.8986e+04, 2.9395e+04,
      2.9809e+04, 3.0230e+04, 3.0656e+04, 3.1088e+04, 3.1527e+04, 3.1971e+04,
      3.2422e+04, 3.2879e+04, 3.3343e+04, 3.3813e+04, 3.4290e+04, 3.4773e+04,
      3.5264e+04, 3.5761e+04, 3.6265e+04, 3.6777e+04, 3.7295e+04, 3.7821e+04,
      3.8354e+04, 3.8895e+04, 3.9444e+04, 4.0000e+04};

  // generated using: 90.0 - numpy.logspace(numpy.log10(90.0 - 4.5e1),
  // numpy.log10(90.0 - 8.5e1), 70) just replace the first element to 0.0;
  const std::vector<REAL> dim1_range = {
      4.5000e+01, 4.6410e+01, 4.7777e+01, 4.9100e+01, 5.0382e+01, 5.1624e+01,
      5.2826e+01, 5.3991e+01, 5.5120e+01, 5.6213e+01, 5.7272e+01, 5.8298e+01,
      5.9292e+01, 6.0254e+01, 6.1186e+01, 6.2089e+01, 6.2964e+01, 6.3812e+01,
      6.4632e+01, 6.5427e+01, 6.6198e+01, 6.6944e+01, 6.7666e+01, 6.8366e+01,
      6.9044e+01, 6.9701e+01, 7.0337e+01, 7.0954e+01, 7.1551e+01, 7.2129e+01,
      7.2689e+01, 7.3231e+01, 7.3757e+01, 7.4266e+01, 7.4759e+01, 7.5237e+01,
      7.5700e+01, 7.6148e+01, 7.6582e+01, 7.7003e+01, 7.7410e+01, 7.7805e+01,
      7.8187e+01, 7.8557e+01, 7.8916e+01, 7.9263e+01, 7.9600e+01, 7.9926e+01,
      8.0241e+01, 8.0547e+01, 8.0843e+01, 8.1130e+01, 8.1408e+01, 8.1678e+01,
      8.1939e+01, 8.2191e+01, 8.2436e+01, 8.2673e+01, 8.2903e+01, 8.3125e+01,
      8.3341e+01, 8.3549e+01, 8.3751e+01, 8.3947e+01, 8.4137e+01, 8.4321e+01,
      8.4499e+01, 8.4671e+01, 8.4838e+01, 8.5000e+01};

  static const inline auto trim_grid_func_0 =
      [](const REAL &dim0_val, const REAL &dim1_val,
         const std::array<REAL, trim_ndim> &rand_nums) {
        std::array<REAL, trim_dim0> result;
        for (int idim = 0; idim < trim_dim0; idim++) {
          result[idim] = (dim0_val * dim1_val);
          result[idim] *=
              rand_nums[0] * Kernel::pow(static_cast<REAL>(idim), 4.0);
        }
        return result;
      };

  static const inline auto trim_grid_func_1 =
      [](const REAL &dim0_val, const REAL &dim1_val,
         const std::array<REAL, trim_ndim> &rand_nums) {
        std::array<REAL, trim_dim0 * trim_dim1> result;
        for (INT counter = 0; counter < trim_dim0 * trim_dim1; counter++) {
          INT idim = static_cast<INT>(static_cast<REAL>(counter) * c1dtd1);
          INT jdim = counter % trim_dim1;
          result[counter] = counter * (dim0_val * dim1_val);
          result[counter] *=
              (rand_nums[0] * Kernel::pow(static_cast<REAL>(idim), 4.0) +
               rand_nums[1] * Kernel::pow(static_cast<REAL>(jdim), 3.0));
        }
        return result;
      };

  static const inline auto trim_grid_func_2 =
      [](const REAL &dim0_val, const REAL &dim1_val,
         const std::array<REAL, trim_ndim> &rand_nums) {
        std::array<REAL, trim_dim0 * trim_dim1 * trim_dim2> result;
        for (INT counter = 0; counter < trim_dim0 * trim_dim1 * trim_dim2;
             counter++) {
          INT idim = static_cast<INT>(static_cast<REAL>(counter) * c1dtd1td2);
          INT jdim =
              static_cast<INT>(static_cast<REAL>(counter) * c1dtd2) % trim_dim1;
          INT kdim = counter % trim_dim2;
          result[counter] = counter * (dim0_val * dim1_val);
          result[counter] *=
              (rand_nums[0] * Kernel::pow(static_cast<REAL>(idim), 4.0) +
               rand_nums[1] * Kernel::pow(static_cast<REAL>(jdim), 3.0) +
               rand_nums[2] * Kernel::pow(static_cast<REAL>(kdim), 2.0));
        }
        return result;
      };

  static const inline auto trim_grid_func_lambda =
      [](const REAL &dim0_val, const REAL &dim1_val,
         const std::array<INT, trim_ndim> &trim_indices,
         const std::array<INT, trim_ndim> &trim_dims,
         const std::array<REAL, trim_ndim> &rand_nums) {
        auto trim_vals_trim_dim0 =
            trim_grid_func_0(dim0_val, dim1_val, rand_nums)[trim_indices[0]];

        auto trim_vals_trim_dim1 = trim_grid_func_1(
            dim0_val, dim1_val,
            rand_nums)[(trim_indices[0] * trim_dims[1]) + trim_indices[1]];

        auto trim_vals_trim_dim2 = trim_grid_func_2(
            dim0_val, dim1_val,
            rand_nums)[(trim_indices[0] * (trim_dims[2] * trim_dims[1])) +
                       (trim_indices[1] * trim_dims[2]) + trim_indices[2]];

        return std::array<REAL, trim_ndim>{
            trim_vals_trim_dim0, trim_vals_trim_dim1, trim_vals_trim_dim2};
      };

  utils::LambdaWrapper<decltype(trim_grid_func_lambda)> trim_grid_func;

public:
  trim_coefficient_values_asym(
      const std::array<REAL, trim_ndim> &rand_nums,
      std::optional<SYCLTargetSharedPtr> sycl_target = std::nullopt)
      : abstract_coefficient_values(), trim_grid_func(trim_grid_func_lambda) {

    this->sycl_target = sycl_target;

    std::vector<std::vector<std::array<REAL, trim_dim0>>> trim_vals_trim_dim0(
        dim1, std::vector<std::array<REAL, trim_dim0>>(dim0));
    std::vector<std::vector<std::array<REAL, trim_dim0 * trim_dim1>>>
        trim_vals_trim_dim1(
            dim1, std::vector<std::array<REAL, trim_dim0 * trim_dim1>>(dim0));
    std::vector<
        std::vector<std::array<REAL, trim_dim0 * trim_dim1 * trim_dim2>>>
        trim_vals_trim_dim2(
            dim1,
            std::vector<std::array<REAL, trim_dim0 * trim_dim1 * trim_dim2>>(
                dim0));

    std::vector<std::vector<std::vector<REAL>>> trim_vals(
        dim1, std::vector<std::vector<REAL>>(dim0));

    for (int idim1 = 0; idim1 < this->dim1; idim1++) {
      for (int idim0 = 0; idim0 < this->dim0; idim0++) {
        trim_vals_trim_dim0[idim1][idim0] = trim_grid_func_0(
            this->dim0_range[idim0], this->dim1_range[idim1], rand_nums);

        trim_vals_trim_dim1[idim1][idim0] = trim_grid_func_1(
            this->dim0_range[idim0], this->dim1_range[idim1], rand_nums);

        trim_vals_trim_dim2[idim1][idim0] = trim_grid_func_2(
            this->dim0_range[idim0], this->dim1_range[idim1], rand_nums);

        for (int itrim_dim0 = 0; itrim_dim0 < trim_dim0; itrim_dim0++) {
          trim_vals[idim1][idim0].push_back(
              trim_vals_trim_dim0[idim1][idim0][itrim_dim0]);
        }

        for (int itrim_dim1 = 0; itrim_dim1 < (trim_dim0 * trim_dim1);
             itrim_dim1++) {
          trim_vals[idim1][idim0].push_back(
              trim_vals_trim_dim1[idim1][idim0][itrim_dim1]);
        }

        for (int itrim_dim2 = 0;
             itrim_dim2 < (trim_dim0 * trim_dim1 * trim_dim2); itrim_dim2++) {
          trim_vals[idim1][idim0].push_back(
              trim_vals_trim_dim2[idim1][idim0][itrim_dim2]);
        }

        this->coeffs_vec.insert(this->coeffs_vec.end(),
                                trim_vals[idim1][idim0].begin(),
                                trim_vals[idim1][idim0].end());
      }
    }

    this->ranges_flat_vec = this->dim0_range;
    this->ranges_flat_vec.insert(this->ranges_flat_vec.end(),
                                 this->dim1_range.begin(),
                                 this->dim1_range.end());

    this->lower_bounds.push_back(this->dim0_range[0]);
    this->lower_bounds.push_back(this->dim1_range[0]);

    this->upper_bounds.push_back(this->dim0_range[this->dim0 - 1]);
    this->upper_bounds.push_back(this->dim1_range[this->dim1 - 1]);

    this->dims_vec.push_back(this->dim0);
    this->dims_vec.push_back(this->dim1);
  }

  auto get_trim_dims_vec() const { return this->trim_dims_vec; }

  auto get_grid_func_data() const {
    std::optional<test_composite_data::TrimEval<ndim + trim_ndim, trim_ndim>>
        return_val;
    if (this->sycl_target) {
      return_val = test_composite_data::TrimEval<ndim + trim_ndim, trim_ndim>(
          this->coeffs_vec, this->ranges_flat_vec, this->dims_vec,
          this->trim_dims_vec, this->sycl_target.value());
    }
    return return_val.value();
  }

  auto get_grid_func() const { return this->trim_grid_func; }
};

struct coefficient_values_3D : abstract_coefficient_values {
private:
  static constexpr int ndim = 3;
  const int dim0 = 8;
  const int dim1 = 10;
  const int dim2 = 15;

  // Generated with python: numpy.linspace(1.0e18, 8.0e18, 8)
  std::vector<REAL> dim0_range = {1.0e+18, 2.0e+18, 3.0e+18, 4.0e+18,
                                  5.0e+18, 6.0e+18, 7.0e+18, 8.0e+18};
  // Generated with python: numpy.logspace(1, 5, 10)
  std::vector<REAL> dim1_range = {
      1.00000000e+01, 2.78255940e+01, 7.74263683e+01, 2.15443469e+02,
      5.99484250e+02, 1.66810054e+03, 4.64158883e+03, 1.29154967e+04,
      3.59381366e+04, 1.00000000e+05};
  // Generated with python: numpy.power(numpy.linspace(1, 15, 15), 2)*1.5 -
  // 100.0
  std::vector<REAL> dim2_range = {-98.5, -94.,  -86.5, -76., -62.5,
                                  -46.,  -26.5, -4.,   21.5, 50.,
                                  81.5,  116.,  153.5, 194., 237.5};

  static constexpr auto grid_func_lambda =
      [](const REAL &dim0_val, const REAL &dim1_val, const REAL &dim2_val) {
        return (dim0_val * dim1_val * dim2_val);
      };

  utils::LambdaWrapper<decltype(grid_func_lambda)> grid_func;

public:
  coefficient_values_3D(
      std::optional<SYCLTargetSharedPtr> sycl_target = std::nullopt)
      : abstract_coefficient_values(), grid_func(grid_func_lambda) {

    this->sycl_target = sycl_target;

    REAL dim2_i = 0.0;
    REAL dim1_i = 0.0;
    REAL dim0_i = 0.0;
    REAL val = 0.0;
    for (int idim2 = 0; idim2 < this->dim2; idim2++) {
      dim2_i = this->dim2_range[idim2];
      for (int idim1 = 0; idim1 < this->dim1; idim1++) {
        dim1_i = this->dim1_range[idim1];
        for (int idim0 = 0; idim0 < this->dim0; idim0++) {
          dim0_i = this->dim0_range[idim0];
          val = this->grid_func(dim0_i, dim1_i, dim2_i);
          this->coeffs_vec.push_back(val);
        }
      }
    }

    this->ranges_flat_vec = this->dim0_range;
    this->ranges_flat_vec.insert(this->ranges_flat_vec.end(),
                                 this->dim1_range.begin(),
                                 this->dim1_range.end());
    this->ranges_flat_vec.insert(this->ranges_flat_vec.end(),
                                 this->dim2_range.begin(),
                                 this->dim2_range.end());

    this->lower_bounds.push_back(this->dim0_range[0]);
    this->lower_bounds.push_back(this->dim1_range[0]);
    this->lower_bounds.push_back(this->dim2_range[0]);

    this->upper_bounds.push_back(this->dim0_range[this->dim0 - 1]);
    this->upper_bounds.push_back(this->dim1_range[this->dim1 - 1]);
    this->upper_bounds.push_back(this->dim2_range[this->dim2 - 1]);

    this->dims_vec.push_back(this->dim0);
    this->dims_vec.push_back(this->dim1);
    this->dims_vec.push_back(this->dim2);
  };

  auto get_grid_func() const { return this->grid_func; }

  auto get_grid_func_data() const {
    return this->get_grid_func_data_ndim<ndim>();
  }
};

struct coefficient_values_4D : abstract_coefficient_values {
private:
  static constexpr int ndim = 4;
  const int dim0 = 8;
  const int dim1 = 10;
  const int dim2 = 15;
  const int dim3 = 23;

  // Generated with python: numpy.linspace(1.0e18, 8.0e18, 8)
  std::vector<REAL> dim0_range = {1.0e+18, 2.0e+18, 3.0e+18, 4.0e+18,
                                  5.0e+18, 6.0e+18, 7.0e+18, 8.0e+18};
  // Generated with python: numpy.logspace(1, 5, 10)
  std::vector<REAL> dim1_range = {
      1.00000000e+01, 2.78255940e+01, 7.74263683e+01, 2.15443469e+02,
      5.99484250e+02, 1.66810054e+03, 4.64158883e+03, 1.29154967e+04,
      3.59381366e+04, 1.00000000e+05};
  // Generated with python: numpy.power(numpy.linspace(1, 15, 15), 2)*1.5 -
  // 100.0
  std::vector<REAL> dim2_range = {-98.5, -94.,  -86.5, -76., -62.5,
                                  -46.,  -26.5, -4.,   21.5, 50.,
                                  81.5,  116.,  153.5, 194., 237.5};
  // Generated with python: numpy.power(numpy.logspace(1, 8, 23), 1.5)
  std::vector<REAL> dim3_range = {
      3.16227766e+01, 9.49014236e+01, 2.84803587e+02, 8.54708813e+02,
      2.56502091e+03, 7.69774706e+03, 2.31012970e+04, 6.93280669e+04,
      2.08056754e+05, 6.24387997e+05, 1.87381742e+06, 5.62341325e+06,
      1.68761248e+07, 5.06460354e+07, 1.51991108e+08, 4.56132386e+08,
      1.36887451e+09, 4.10805608e+09, 1.23284674e+10, 3.69983041e+10,
      1.11033632e+11, 3.33217094e+11, 1.00000000e+12};

  static constexpr auto grid_func_lambda =
      [](const REAL &dim0_val, const REAL &dim1_val, const REAL &dim2_val,
         const REAL &dim3_val) {
        return (dim0_val * dim1_val * dim2_val * dim3_val);
      };

  utils::LambdaWrapper<decltype(grid_func_lambda)> grid_func;

public:
  coefficient_values_4D(
      std::optional<SYCLTargetSharedPtr> sycl_target = std::nullopt)
      : abstract_coefficient_values(), grid_func(grid_func_lambda) {

    this->sycl_target = sycl_target;

    REAL dim3_i = 0.0;
    REAL dim2_i = 0.0;
    REAL dim1_i = 0.0;
    REAL dim0_i = 0.0;
    REAL val = 0.0;
    for (int idim3 = 0; idim3 < this->dim3; idim3++) {
      dim3_i = this->dim3_range[idim3];
      for (int idim2 = 0; idim2 < this->dim2; idim2++) {
        dim2_i = this->dim2_range[idim2];
        for (int idim1 = 0; idim1 < this->dim1; idim1++) {
          dim1_i = this->dim1_range[idim1];
          for (int idim0 = 0; idim0 < this->dim0; idim0++) {
            dim0_i = this->dim0_range[idim0];
            val = this->grid_func(dim0_i, dim1_i, dim2_i, dim3_i);
            this->coeffs_vec.push_back(val);
          }
        }
      }
    }

    this->ranges_flat_vec = this->dim0_range;
    this->ranges_flat_vec.insert(this->ranges_flat_vec.end(),
                                 this->dim1_range.begin(),
                                 this->dim1_range.end());
    this->ranges_flat_vec.insert(this->ranges_flat_vec.end(),
                                 this->dim2_range.begin(),
                                 this->dim2_range.end());
    this->ranges_flat_vec.insert(this->ranges_flat_vec.end(),
                                 this->dim3_range.begin(),
                                 this->dim3_range.end());

    this->lower_bounds.push_back(this->dim0_range[0]);
    this->lower_bounds.push_back(this->dim1_range[0]);
    this->lower_bounds.push_back(this->dim2_range[0]);
    this->lower_bounds.push_back(this->dim3_range[0]);

    this->upper_bounds.push_back(this->dim0_range[this->dim0 - 1]);
    this->upper_bounds.push_back(this->dim1_range[this->dim1 - 1]);
    this->upper_bounds.push_back(this->dim2_range[this->dim2 - 1]);
    this->upper_bounds.push_back(this->dim3_range[this->dim3 - 1]);

    this->dims_vec.push_back(this->dim0);
    this->dims_vec.push_back(this->dim1);
    this->dims_vec.push_back(this->dim2);
    this->dims_vec.push_back(this->dim3);
  };

  auto get_grid_func() const { return this->grid_func; }

  auto get_grid_func_data() const {
    return this->get_grid_func_data_ndim<ndim>();
  }
};

struct coefficient_values_5D : abstract_coefficient_values {
private:
  static constexpr int ndim = 5;
  const int dim0 = 8;
  const int dim1 = 10;
  const int dim2 = 15;
  const int dim3 = 23;
  const int dim4 = 35;

  // Generated with python: numpy.linspace(1.0e18, 8.0e18, 8)
  std::vector<REAL> dim0_range = {1.0e+18, 2.0e+18, 3.0e+18, 4.0e+18,
                                  5.0e+18, 6.0e+18, 7.0e+18, 8.0e+18};
  // Generated with python: numpy.logspace(1, 5, 10)
  std::vector<REAL> dim1_range = {
      1.00000000e+01, 2.78255940e+01, 7.74263683e+01, 2.15443469e+02,
      5.99484250e+02, 1.66810054e+03, 4.64158883e+03, 1.29154967e+04,
      3.59381366e+04, 1.00000000e+05};
  // Generated with python: numpy.power(numpy.linspace(1, 15, 15), 2)*1.5 -
  // 100.0
  std::vector<REAL> dim2_range = {-98.5, -94.,  -86.5, -76., -62.5,
                                  -46.,  -26.5, -4.,   21.5, 50.,
                                  81.5,  116.,  153.5, 194., 237.5};
  // Generated with python: numpy.power(numpy.logspace(1, 8, 23), 1.5)
  std::vector<REAL> dim3_range = {
      3.16227766e+01, 9.49014236e+01, 2.84803587e+02, 8.54708813e+02,
      2.56502091e+03, 7.69774706e+03, 2.31012970e+04, 6.93280669e+04,
      2.08056754e+05, 6.24387997e+05, 1.87381742e+06, 5.62341325e+06,
      1.68761248e+07, 5.06460354e+07, 1.51991108e+08, 4.56132386e+08,
      1.36887451e+09, 4.10805608e+09, 1.23284674e+10, 3.69983041e+10,
      1.11033632e+11, 3.33217094e+11, 1.00000000e+12};
  // Generated with python: numpy.power(numpy.logspace(1, 5, 35), 0.1)
  std::vector<REAL> dim4_range = {
      1.25892541, 1.29349486, 1.32901356, 1.36550759, 1.40300372, 1.44152948,
      1.48111314, 1.52178375, 1.56357114, 1.606506,   1.65061983, 1.695945,
      1.74251478, 1.79036334, 1.8395258,  1.89003823, 1.9419377,  1.99526231,
      2.05005119, 2.10634454, 2.16418368, 2.22361105, 2.28467027, 2.34740614,
      2.4118647,  2.47809326, 2.54614043, 2.61605614, 2.68789169, 2.76169981,
      2.83753467, 2.91545191, 2.99550872, 3.07776385, 3.16227766};

  static constexpr auto grid_func_lambda =
      [](const REAL &dim0_val, const REAL &dim1_val, const REAL &dim2_val,
         const REAL &dim3_val, const REAL &dim4_val) {
        return (dim0_val * dim1_val * dim2_val * dim3_val * dim4_val);
      };

  utils::LambdaWrapper<decltype(grid_func_lambda)> grid_func;

public:
  coefficient_values_5D(
      std::optional<SYCLTargetSharedPtr> sycl_target = std::nullopt)
      : abstract_coefficient_values(), grid_func(grid_func_lambda) {

    this->sycl_target = sycl_target;

    REAL dim4_i = 0.0;
    REAL dim3_i = 0.0;
    REAL dim2_i = 0.0;
    REAL dim1_i = 0.0;
    REAL dim0_i = 0.0;
    REAL val = 0.0;
    for (int idim4 = 0; idim4 < this->dim4; idim4++) {
      dim4_i = this->dim4_range[idim4];
      for (int idim3 = 0; idim3 < this->dim3; idim3++) {
        dim3_i = this->dim3_range[idim3];
        for (int idim2 = 0; idim2 < this->dim2; idim2++) {
          dim2_i = this->dim2_range[idim2];
          for (int idim1 = 0; idim1 < this->dim1; idim1++) {
            dim1_i = this->dim1_range[idim1];
            for (int idim0 = 0; idim0 < this->dim0; idim0++) {
              dim0_i = this->dim0_range[idim0];
              val = this->grid_func(dim0_i, dim1_i, dim2_i, dim3_i, dim4_i);
              this->coeffs_vec.push_back(val);
            }
          }
        }
      }
    }

    this->ranges_flat_vec = this->dim0_range;
    this->ranges_flat_vec.insert(ranges_flat_vec.end(),
                                 this->dim1_range.begin(),
                                 this->dim1_range.end());
    this->ranges_flat_vec.insert(ranges_flat_vec.end(),
                                 this->dim2_range.begin(),
                                 this->dim2_range.end());
    this->ranges_flat_vec.insert(ranges_flat_vec.end(),
                                 this->dim3_range.begin(),
                                 this->dim3_range.end());
    this->ranges_flat_vec.insert(ranges_flat_vec.end(),
                                 this->dim4_range.begin(),
                                 this->dim4_range.end());

    this->lower_bounds.push_back(this->dim0_range[0]);
    this->lower_bounds.push_back(this->dim1_range[0]);
    this->lower_bounds.push_back(this->dim2_range[0]);
    this->lower_bounds.push_back(this->dim3_range[0]);
    this->lower_bounds.push_back(this->dim4_range[0]);

    this->upper_bounds.push_back(this->dim0_range[this->dim0 - 1]);
    this->upper_bounds.push_back(this->dim1_range[this->dim1 - 1]);
    this->upper_bounds.push_back(this->dim2_range[this->dim2 - 1]);
    this->upper_bounds.push_back(this->dim3_range[this->dim3 - 1]);
    this->upper_bounds.push_back(this->dim4_range[this->dim4 - 1]);

    this->dims_vec.push_back(this->dim0);
    this->dims_vec.push_back(this->dim1);
    this->dims_vec.push_back(this->dim2);
    this->dims_vec.push_back(this->dim3);
    this->dims_vec.push_back(this->dim4);
  };

  auto get_grid_func() const { return this->grid_func; }

  auto get_grid_func_data() const {
    return this->get_grid_func_data_ndim<ndim>();
  }
};

#endif
