#ifndef REACTIONS_MOCK_INTERPOLATION_DATA_H
#define REACTIONS_MOCK_INTERPOLATION_DATA_H
#include <memory>
#include <neso_particles.hpp>
#include <reactions/reactions.hpp>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

namespace test_composite_data {
template <int input_ndim>
struct GridEvalOnDevice
    : public ReactionDataBaseOnDevice<1, DEFAULT_RNG_KERNEL, input_ndim> {
  GridEvalOnDevice() = default;

  std::array<REAL, 1> calc_data(
      const std::array<REAL, input_ndim> &input,
      [[maybe_unused]] const Access::LoopIndex::Read &index,
      [[maybe_unused]] const Access::SymVector::Write<INT> &req_int_props,
      [[maybe_unused]] const Access::SymVector::Read<REAL> &req_real_props,
      [[maybe_unused]] DEFAULT_RNG_KERNEL::KernelType &rng_kernel) const {
    std::array<INT, input_ndim> grid_indices;
    grid_indices[0] = interp_utils::calc_floor_point_index(
        input[0], this->d_ranges, this->d_dims[0]);
    size_t aggregate_dims = 0;
    for (size_t i = 1; i < input_ndim; i++) {
      aggregate_dims += this->d_dims[i - 1];
      grid_indices[i] = interp_utils::calc_floor_point_index(
          input[i], this->d_ranges + aggregate_dims, this->d_dims[i]);
    }

    auto grid_indices_ptr = grid_indices.data();
    INT grid_flat_index = interp_utils::coeff_index_on_device(
        grid_indices_ptr, this->d_dims, input_ndim);

    return std::array<REAL, 1>{this->d_grid[grid_flat_index]};
  }

public:
  size_t const *d_dims;
  REAL const *d_ranges;
  REAL const *d_grid;
};

template <int input_ndim>
struct GridEval : public ReactionDataBase<GridEvalOnDevice<input_ndim>> {
  GridEval(const std::vector<REAL> &grid, const std::vector<REAL> &ranges_vec,
           const std::vector<size_t> &dims_vec,
           SYCLTargetSharedPtr sycl_target) {
    this->on_device_obj = GridEvalOnDevice<input_ndim>();

    this->h_dims =
        std::make_shared<BufferDevice<size_t>>(sycl_target, dims_vec);
    this->on_device_obj->d_dims = this->h_dims->ptr;

    this->h_ranges =
        std::make_shared<BufferDevice<REAL>>(sycl_target, ranges_vec);
    this->on_device_obj->d_ranges = this->h_ranges->ptr;

    this->h_grid = std::make_shared<BufferDevice<REAL>>(sycl_target, grid);
    this->on_device_obj->d_grid = this->h_grid->ptr;
  };

public:
  std::shared_ptr<BufferDevice<size_t>> h_dims;
  std::shared_ptr<BufferDevice<REAL>> h_ranges;
  std::shared_ptr<BufferDevice<REAL>> h_grid;
};

template <int input_ndim, int output_ndim>
struct TrimEvalOnDevice
    : public ReactionDataBaseOnDevice<output_ndim, DEFAULT_RNG_KERNEL,
                                      input_ndim, REAL, INT> {
  TrimEvalOnDevice() {
    static_assert(
        input_ndim >= output_ndim,
        "For TrimEvalOnDevice, input_ndim >= output_ndim must be true.");
  };

  std::array<REAL, output_ndim> calc_data(
      const std::array<REAL, input_ndim> &input,
      [[maybe_unused]] const Access::LoopIndex::Read &index,
      [[maybe_unused]] const Access::SymVector::Write<INT> &req_int_props,
      [[maybe_unused]] const Access::SymVector::Read<REAL> &req_real_props,
      [[maybe_unused]] DEFAULT_RNG_KERNEL::KernelType &rng_kernel) const {

    std::array<REAL, output_ndim> input_to_bin;
    std::array<INT, output_ndim> trim_dims_arr;
    for (size_t i = 0; i < output_ndim; i++) {
      input_to_bin[i] = input[i + interp_ndim];
      trim_dims_arr[i] = this->d_trim_dims[i];
    }

    std::array<INT, output_ndim> binned_inputs =
        interp_utils::bin_uniform_indices(input_to_bin, trim_dims_arr);

    std::array<INT, interp_ndim> grid_indices;
    grid_indices[0] = interp_utils::calc_floor_point_index(
        input[0], this->d_ranges, this->d_dims[0]);
    size_t aggregate_dims = 0;
    for (size_t i = 1; i < interp_ndim; i++) {
      aggregate_dims += this->d_dims[i - 1];
      grid_indices[i] = interp_utils::calc_floor_point_index(
          input[i], this->d_ranges + aggregate_dims, this->d_dims[i]);
    }

    auto grid_indices_ptr = grid_indices.data();
    INT grid_flat_index =
        interp_utils::coeff_index_on_device(grid_indices_ptr, this->d_dims, 2);

    auto grid_access_point = grid_flat_index * this->grid_stride;

    std::array<INT, output_ndim> trim_indices;
    std::array<REAL, output_ndim> trim_vals;

    int field_access_point;
    int field_stride;
    int aggregate_dim;
    for (int idim = 0; idim < output_ndim; idim++) {
      field_access_point = 0;
      field_stride = 0;
      aggregate_dim = 1;
      for (int jdim = idim - 1, kdim = 0; jdim >= 0; jdim--, kdim++) {
        aggregate_dim *= d_trim_dims[kdim];
        field_access_point += aggregate_dim;
        field_stride += (binned_inputs[jdim] * aggregate_dim);
      }

      trim_indices[idim] =
          field_access_point + field_stride + binned_inputs[idim];
      trim_vals[idim] = this->d_grid[grid_access_point + trim_indices[idim]];
    }
    return trim_vals;
  }

public:
  int grid_stride;

  static constexpr int interp_ndim = input_ndim - output_ndim;

  REAL const *d_ranges;
  size_t const *d_dims;
  size_t const *d_trim_dims;
  REAL const *d_grid;
};

template <int input_ndim, int output_ndim>
struct TrimEval
    : public ReactionDataBase<TrimEvalOnDevice<input_ndim, output_ndim>> {
  TrimEval(const std::vector<REAL> &grid, const std::vector<REAL> &ranges_vec,
           const std::vector<size_t> &dims_vec,
           const std::vector<size_t> &trim_dims_vec,
           SYCLTargetSharedPtr sycl_target) {
    this->on_device_obj = TrimEvalOnDevice<input_ndim, output_ndim>();

    this->h_grid = std::make_shared<BufferDevice<REAL>>(sycl_target, grid);
    this->on_device_obj->d_grid = this->h_grid->ptr;

    this->h_ranges =
        std::make_shared<BufferDevice<REAL>>(sycl_target, ranges_vec);
    this->on_device_obj->d_ranges = this->h_ranges->ptr;

    this->h_dims =
        std::make_shared<BufferDevice<size_t>>(sycl_target, dims_vec);
    this->on_device_obj->d_dims = this->h_dims->ptr;

    int aggregrate_dim = 1;
    this->on_device_obj->grid_stride = 0;
    for (auto &trim_dim : trim_dims_vec) {
      aggregrate_dim *= trim_dim;
      this->on_device_obj->grid_stride += aggregrate_dim;
    }

    this->h_trim_dims =
        std::make_shared<BufferDevice<size_t>>(sycl_target, trim_dims_vec);
    this->on_device_obj->d_trim_dims = this->h_trim_dims->ptr;
  };

public:
  std::shared_ptr<BufferDevice<REAL>> h_ranges;
  std::shared_ptr<BufferDevice<size_t>> h_dims;
  std::shared_ptr<BufferDevice<size_t>> h_trim_dims;
  std::shared_ptr<BufferDevice<REAL>> h_grid;
};

}; // namespace test_composite_data

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

  static constexpr int c1dtd1 = 1.0 / trim_dim1;
  static constexpr int c1dtd2 = 1.0 / trim_dim2;
  static constexpr int c1dtd1td2 = 1.0 / (trim_dim1 * trim_dim2);

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
          INT idim = counter * c1dtd1;
          INT jdim = counter % trim_dim1;
          result[counter] = (dim0_val * dim1_val);
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
          INT idim = counter * c1dtd1td2;
          INT jdim = (counter * c1dtd2) % trim_dim1;
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
