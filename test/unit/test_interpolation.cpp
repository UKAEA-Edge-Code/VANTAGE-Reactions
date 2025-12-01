#include "include/mock_particle_group.hpp"
#include "reactions_lib/reaction_data/interpolate_data.hpp"
#include <array>
#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <memory>
#include <neso_particles.hpp>
#include <reactions/reactions.hpp>
#include <vector>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

struct coefficient_values_1D {
private:
  static constexpr int ndim = 1;
  static constexpr size_t dim0 = 8;

  // Generated with python: numpy.linspace(1.0e18, 8.0e18, 8)
  static constexpr std::array<REAL, dim0> dim0_range = {
      1.0e+18, 2.0e+18, 3.0e+18, 4.0e+18, 5.0e+18, 6.0e+18, 7.0e+18, 8.0e+18};

  std::vector<REAL> coeffs_vec;
  std::vector<REAL> ranges_flat_vec;

public:
  coefficient_values_1D() {
    for (int idim0 = 0; idim0 < dim0; idim0++) {
      this->coeffs_vec.push_back(std::log10(this->dim0_range[idim0]));
    }

    std::vector<REAL> dim0_range_vec(this->dim0_range.begin(),
                                     this->dim0_range.end());

    std::vector<REAL> ranges_flat_vec = dim0_range_vec;
    this->ranges_flat_vec = ranges_flat_vec;
  };

  const std::vector<REAL> &get_coeffs_vec() { return this->coeffs_vec; }

  const std::vector<REAL> &get_ranges_flat_vec() {
    return this->ranges_flat_vec;
  }

  std::vector<size_t> get_dims_vec() { return std::vector<size_t>{this->dim0}; }
};

struct coefficient_values_2D {
private:
  static constexpr int ndim = 2;
  static const size_t ndens_dim = 8;
  static const size_t ntemp_dim = 10;

  // Generated with python: numpy.linspace(1.0e18, 8.0e18, 8)
  static constexpr std::array<REAL, ndens_dim> dens_range = {
      1.0e+18, 2.0e+18, 3.0e+18, 4.0e+18, 5.0e+18, 6.0e+18, 7.0e+18, 8.0e+18};
  // Generated with python: numpy.logspace(1, 5, 10)
  static constexpr std::array<REAL, ntemp_dim> temp_range = {
      1.00000000e+01, 2.78255940e+01, 7.74263683e+01, 2.15443469e+02,
      5.99484250e+02, 1.66810054e+03, 4.64158883e+03, 1.29154967e+04,
      3.59381366e+04, 1.00000000e+05};

  std::vector<REAL> coeffs_vec;
  std::vector<REAL> ranges_flat_vec;

public:
  coefficient_values_2D() {
    REAL temp_i = 0.0;
    for (int itemp = 0; itemp < ntemp_dim; itemp++) {
      temp_i = this->temp_range[itemp];
      for (int idens = 0; idens < ndens_dim; idens++) {
        this->coeffs_vec.push_back(temp_i * this->dens_range[idens]);
      }
    }

    std::vector<REAL> dens_range_vec(this->dens_range.begin(),
                                     this->dens_range.end());
    std::vector<REAL> temp_range_vec(this->temp_range.begin(),
                                     this->temp_range.end());

    std::vector<REAL> ranges_flat_vec = dens_range_vec;
    ranges_flat_vec.insert(ranges_flat_vec.end(), temp_range_vec.begin(),
                           temp_range_vec.end());
    this->ranges_flat_vec = ranges_flat_vec;
  };

  const std::vector<REAL> &get_coeffs_vec() { return this->coeffs_vec; }

  const std::vector<REAL> &get_ranges_flat_vec() {
    return this->ranges_flat_vec;
  }

  std::vector<size_t> get_dims_vec() {
    return std::vector<size_t>{this->ndens_dim, this->ntemp_dim};
  }
};

struct coefficient_values_3D {
private:
  static constexpr int ndim = 3;
  static constexpr size_t dim0 = 8;
  static constexpr size_t dim1 = 10;
  static constexpr size_t dim2 = 15;

  // Generated with python: numpy.linspace(1.0e18, 8.0e18, 8)
  static constexpr std::array<REAL, dim0> dim0_range = {
      1.0e+18, 2.0e+18, 3.0e+18, 4.0e+18, 5.0e+18, 6.0e+18, 7.0e+18, 8.0e+18};
  // Generated with python: numpy.logspace(1, 5, 10)
  static constexpr std::array<REAL, dim1> dim1_range = {
      1.00000000e+01, 2.78255940e+01, 7.74263683e+01, 2.15443469e+02,
      5.99484250e+02, 1.66810054e+03, 4.64158883e+03, 1.29154967e+04,
      3.59381366e+04, 1.00000000e+05};
  // Generated with python: numpy.power(np.linspace(1, 15, 15), 2)*1.5 - 100.0
  static constexpr std::array<REAL, dim2> dim2_range = {
      -98.5, -94., -86.5, -76., -62.5, -46., -26.5, -4.,
      21.5,  50.,  81.5,  116., 153.5, 194., 237.5};

  std::vector<REAL> coeffs_vec;
  std::vector<REAL> ranges_flat_vec;

public:
  coefficient_values_3D() {
    REAL dim2_i = 0.0;
    REAL dim1_i = 0.0;
    for (int idim2 = 0; idim2 < dim2; idim2++) {
      dim2_i = this->dim2_range[idim2];
      for (int idim1 = 0; idim1 < dim1; idim1++) {
        dim1_i = this->dim1_range[idim1];
        for (int idim0 = 0; idim0 < dim0; idim0++) {
          this->coeffs_vec.push_back(dim2_i * dim1_i * this->dim0_range[idim0]);
        }
      }
    }

    std::vector<REAL> dim0_range_vec(this->dim0_range.begin(),
                                     this->dim0_range.end());
    std::vector<REAL> dim1_range_vec(this->dim1_range.begin(),
                                     this->dim1_range.end());
    std::vector<REAL> dim2_range_vec(this->dim2_range.begin(),
                                     this->dim2_range.end());

    std::vector<REAL> ranges_flat_vec = dim0_range_vec;
    ranges_flat_vec.insert(ranges_flat_vec.end(), dim1_range_vec.begin(),
                           dim1_range_vec.end());
    ranges_flat_vec.insert(ranges_flat_vec.end(), dim2_range_vec.begin(),
                           dim2_range_vec.end());
    this->ranges_flat_vec = ranges_flat_vec;
  };

  const std::vector<REAL> &get_coeffs_vec() { return this->coeffs_vec; }

  const std::vector<REAL> &get_ranges_flat_vec() {
    return this->ranges_flat_vec;
  }

  std::vector<size_t> get_dims_vec() {
    return std::vector<size_t>{this->dim0, this->dim1, this->dim2};
  }
};

struct coefficient_values_4D {
private:
  static constexpr int ndim = 4;
  static constexpr size_t dim0 = 8;
  static constexpr size_t dim1 = 10;
  static constexpr size_t dim2 = 15;
  static constexpr size_t dim3 = 23;

  // Generated with python: numpy.linspace(1.0e18, 8.0e18, 8)
  static constexpr std::array<REAL, dim0> dim0_range = {
      1.0e+18, 2.0e+18, 3.0e+18, 4.0e+18, 5.0e+18, 6.0e+18, 7.0e+18, 8.0e+18};
  // Generated with python: numpy.logspace(1, 5, 10)
  static constexpr std::array<REAL, dim1> dim1_range = {
      1.00000000e+01, 2.78255940e+01, 7.74263683e+01, 2.15443469e+02,
      5.99484250e+02, 1.66810054e+03, 4.64158883e+03, 1.29154967e+04,
      3.59381366e+04, 1.00000000e+05};
  // Generated with python: numpy.power(numpy.linspace(1, 15, 15), 2)*1.5 -
  // 100.0
  static constexpr std::array<REAL, dim2> dim2_range = {
      -98.5, -94., -86.5, -76., -62.5, -46., -26.5, -4.,
      21.5,  50.,  81.5,  116., 153.5, 194., 237.5};
  // Generated with python: numpy.power(numpy.logspace(1, 8, 23), 1.5)
  static constexpr std::array<REAL, dim3> dim3_range = {
      3.16227766e+01, 9.49014236e+01, 2.84803587e+02, 8.54708813e+02,
      2.56502091e+03, 7.69774706e+03, 2.31012970e+04, 6.93280669e+04,
      2.08056754e+05, 6.24387997e+05, 1.87381742e+06, 5.62341325e+06,
      1.68761248e+07, 5.06460354e+07, 1.51991108e+08, 4.56132386e+08,
      1.36887451e+09, 4.10805608e+09, 1.23284674e+10, 3.69983041e+10,
      1.11033632e+11, 3.33217094e+11, 1.00000000e+12};

  std::vector<REAL> coeffs_vec;
  std::vector<REAL> ranges_flat_vec;

public:
  coefficient_values_4D() {
    REAL dim3_i = 0.0;
    REAL dim2_i = 0.0;
    REAL dim1_i = 0.0;
    for (int idim3 = 0; idim3 < dim3; idim3++) {
      dim3_i = this->dim3_range[idim3];
      for (int idim2 = 0; idim2 < dim2; idim2++) {
        dim2_i = this->dim2_range[idim2];
        for (int idim1 = 0; idim1 < dim1; idim1++) {
          dim1_i = this->dim1_range[idim1];
          for (int idim0 = 0; idim0 < dim0; idim0++) {
            this->coeffs_vec.push_back(dim3_i * dim2_i * dim1_i *
                                       this->dim0_range[idim0]);
          }
        }
      }
    }

    std::vector<REAL> dim0_range_vec(this->dim0_range.begin(),
                                     this->dim0_range.end());
    std::vector<REAL> dim1_range_vec(this->dim1_range.begin(),
                                     this->dim1_range.end());
    std::vector<REAL> dim2_range_vec(this->dim2_range.begin(),
                                     this->dim2_range.end());
    std::vector<REAL> dim3_range_vec(this->dim3_range.begin(),
                                     this->dim3_range.end());

    std::vector<REAL> ranges_flat_vec = dim0_range_vec;
    ranges_flat_vec.insert(ranges_flat_vec.end(), dim1_range_vec.begin(),
                           dim1_range_vec.end());
    ranges_flat_vec.insert(ranges_flat_vec.end(), dim2_range_vec.begin(),
                           dim2_range_vec.end());
    ranges_flat_vec.insert(ranges_flat_vec.end(), dim3_range_vec.begin(),
                           dim3_range_vec.end());
    this->ranges_flat_vec = ranges_flat_vec;
  };

  const std::vector<REAL> &get_coeffs_vec() { return this->coeffs_vec; }

  const std::vector<REAL> &get_ranges_flat_vec() {
    return this->ranges_flat_vec;
  }

  std::vector<size_t> get_dims_vec() {
    return std::vector<size_t>{this->dim0, this->dim1, this->dim2, this->dim3};
  }
};

struct coefficient_values_5D {
private:
  static constexpr int ndim = 5;
  static constexpr size_t dim0 = 8;
  static constexpr size_t dim1 = 10;
  static constexpr size_t dim2 = 15;
  static constexpr size_t dim3 = 23;
  static constexpr size_t dim4 = 35;

  // Generated with python: numpy.linspace(1.0e18, 8.0e18, 8)
  static constexpr std::array<REAL, dim0> dim0_range = {
      1.0e+18, 2.0e+18, 3.0e+18, 4.0e+18, 5.0e+18, 6.0e+18, 7.0e+18, 8.0e+18};
  // Generated with python: numpy.logspace(1, 5, 10)
  static constexpr std::array<REAL, dim1> dim1_range = {
      1.00000000e+01, 2.78255940e+01, 7.74263683e+01, 2.15443469e+02,
      5.99484250e+02, 1.66810054e+03, 4.64158883e+03, 1.29154967e+04,
      3.59381366e+04, 1.00000000e+05};
  // Generated with python: numpy.power(numpy.linspace(1, 15, 15), 2)*1.5 -
  // 100.0
  static constexpr std::array<REAL, dim2> dim2_range = {
      -98.5, -94., -86.5, -76., -62.5, -46., -26.5, -4.,
      21.5,  50.,  81.5,  116., 153.5, 194., 237.5};
  // Generated with python: numpy.power(numpy.logspace(1, 8, 23), 1.5)
  static constexpr std::array<REAL, dim3> dim3_range = {
      3.16227766e+01, 9.49014236e+01, 2.84803587e+02, 8.54708813e+02,
      2.56502091e+03, 7.69774706e+03, 2.31012970e+04, 6.93280669e+04,
      2.08056754e+05, 6.24387997e+05, 1.87381742e+06, 5.62341325e+06,
      1.68761248e+07, 5.06460354e+07, 1.51991108e+08, 4.56132386e+08,
      1.36887451e+09, 4.10805608e+09, 1.23284674e+10, 3.69983041e+10,
      1.11033632e+11, 3.33217094e+11, 1.00000000e+12};
  // Generated with python: numpy.power(numpy.logspace(1, 5, 35), 0.1)
  static constexpr std::array<REAL, dim4> dim4_range = {
      1.25892541, 1.29349486, 1.32901356, 1.36550759, 1.40300372, 1.44152948,
      1.48111314, 1.52178375, 1.56357114, 1.606506,   1.65061983, 1.695945,
      1.74251478, 1.79036334, 1.8395258,  1.89003823, 1.9419377,  1.99526231,
      2.05005119, 2.10634454, 2.16418368, 2.22361105, 2.28467027, 2.34740614,
      2.4118647,  2.47809326, 2.54614043, 2.61605614, 2.68789169, 2.76169981,
      2.83753467, 2.91545191, 2.99550872, 3.07776385, 3.16227766};

  std::vector<REAL> coeffs_vec;
  std::vector<REAL> ranges_flat_vec;

public:
  coefficient_values_5D() {
    REAL dim4_i = 0.0;
    REAL dim3_i = 0.0;
    REAL dim2_i = 0.0;
    REAL dim1_i = 0.0;
    for (int idim4 = 0; idim4 < dim4; idim4++) {
      dim4_i = this->dim4_range[idim4];
      for (int idim3 = 0; idim3 < dim3; idim3++) {
        dim3_i = this->dim3_range[idim3];
        for (int idim2 = 0; idim2 < dim2; idim2++) {
          dim2_i = this->dim2_range[idim2];
          for (int idim1 = 0; idim1 < dim1; idim1++) {
            dim1_i = this->dim1_range[idim1];
            for (int idim0 = 0; idim0 < dim0; idim0++) {
              this->coeffs_vec.push_back(dim4_i * dim3_i * dim2_i * dim1_i *
                                         this->dim0_range[idim0]);
            }
          }
        }
      }
    }

    std::vector<REAL> dim0_range_vec(this->dim0_range.begin(),
                                     this->dim0_range.end());
    std::vector<REAL> dim1_range_vec(this->dim1_range.begin(),
                                     this->dim1_range.end());
    std::vector<REAL> dim2_range_vec(this->dim2_range.begin(),
                                     this->dim2_range.end());
    std::vector<REAL> dim3_range_vec(this->dim3_range.begin(),
                                     this->dim3_range.end());
    std::vector<REAL> dim4_range_vec(this->dim4_range.begin(),
                                     this->dim4_range.end());

    std::vector<REAL> ranges_flat_vec = dim0_range_vec;
    ranges_flat_vec.insert(ranges_flat_vec.end(), dim1_range_vec.begin(),
                           dim1_range_vec.end());
    ranges_flat_vec.insert(ranges_flat_vec.end(), dim2_range_vec.begin(),
                           dim2_range_vec.end());
    ranges_flat_vec.insert(ranges_flat_vec.end(), dim3_range_vec.begin(),
                           dim3_range_vec.end());
    ranges_flat_vec.insert(ranges_flat_vec.end(), dim4_range_vec.begin(),
                           dim4_range_vec.end());
    this->ranges_flat_vec = ranges_flat_vec;
  };

  const std::vector<REAL> &get_coeffs_vec() { return this->coeffs_vec; }

  const std::vector<REAL> &get_ranges_flat_vec() {
    return this->ranges_flat_vec;
  }

  std::vector<size_t> get_dims_vec() {
    return std::vector<size_t>{this->dim0, this->dim1, this->dim2, this->dim3,
                               this->dim4};
  }
};

inline void
diagnostic_output(const int &particle_count_, const int &dim_index_,
                  const int &num_points_,
                  Access::LocalMemoryInterlaced::Write<size_t> origin_indices_,
                  Access::LocalMemoryInterlaced::Write<int> vertices_,
                  Access::LocalMemoryInterlaced::Write<REAL> func_evals_,
                  size_t *dims_vec_ptr_, REAL *ranges_vec_ptr_) {
  for (int i = 0; i < num_points_; i++) {
    if (particle_count_ == 0) {
      for (int j = dim_index_; j >= 0; j--) {
        size_t temp_binary_repr =
            origin_indices_.at(j) + binary_extract(vertices_.at(i), j);
        REAL ranges_val = ranges_vec_ptr_[interp_utils::range_index_on_device(
            temp_binary_repr, size_t(j), dims_vec_ptr_)];
        printf("%ld, %e\t", temp_binary_repr, ranges_val);
      }
      printf("%e\n", func_evals_.at(i));
    }
  }
  if (particle_count_ == 0) {
    printf("\n");
  }
};

TEST(InterpolationTest, REACTION_DATA_1D) {
  // Interpolation points
  REAL prop_interp_0 = 6.4e18;
  REAL expected_interp_value = std::log10(prop_interp_0);
  static constexpr int ndim = 1;

  auto particle_group = create_test_particle_group(1e5);

  auto sycl_target = particle_group->sycl_target;

  auto npart = particle_group->get_npart_local();

  // This is a bit of a workaround for the fact that currently I can't use
  // PipelineData to pass a ConcatenatorData object that contains data from
  // multiple 1D ExtractorData objects and pass that to an InterpolateData
  // object.
  particle_group->add_particle_dat(Sym<REAL>("COMBINED_PROP"), ndim);

  particle_loop(
      particle_group, [=](auto prop) { prop.at(0) = prop_interp_0; },
      Access::write(Sym<REAL>("COMBINED_PROP")))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  // Setup the mock ADAS data values.
  auto ADAS_values = coefficient_values_1D();
  auto dims_vec = ADAS_values.get_dims_vec();
  auto ranges_vec = ADAS_values.get_ranges_flat_vec();
  auto grid = ADAS_values.get_coeffs_vec();

  // The input and output ndims being 2  for the InterpolateData (for this
  // specific unit test) is because PipelineData only presents the output ndims
  // of the last object in the pipeline when DataCalculator queries the
  // ndims of the passed PipelineData object. This is fine if either all objects
  // in the pipeline have the same output ndims OR if the last output ndims is
  // greater than the output ndims of any preceeding object in the pipeline.
  // For the current implementation of InterpolateData, I would
  // prefer input ndim be equal to the dimensionality of the grid.
  // and output ndim being 1 but this works fine for testing. (I'm just storing
  // the result in the 0th index of the array returned by
  // InterpolateData.calc_data and test against that).
  auto extractor_data = ExtractorData<ndim>(Sym<REAL>("COMBINED_PROP"));
  auto interpolator_data =
      InterpolateData<ndim>(dims_vec, ranges_vec, grid, sycl_target);
  auto interpolator_data_on_device = interpolator_data.get_on_device_obj();

  auto req_int_props_ = interpolator_data.get_required_int_sym_vector();
  auto req_real_props_ = interpolator_data.get_required_real_sym_vector();

  auto extractor_data_calc =
      DataCalculator<decltype(extractor_data)>(extractor_data);

  auto pre_req_extract_data = std::make_shared<NDLocalArray<REAL, 2>>(
      sycl_target, 0, extractor_data_calc.get_data_size());
  pre_req_extract_data->fill(0);

  auto pre_req_interpolator_data = std::make_shared<NDLocalArray<REAL, 2>>(
      sycl_target, 0, interpolator_data.get_dim());
  pre_req_interpolator_data->fill(0);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = pre_req_extract_data->index.shape;
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    pre_req_extract_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape[1]);
    pre_req_extract_data->fill(0);

    shape = pre_req_interpolator_data->index.shape;
    pre_req_interpolator_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape[1]);
    pre_req_interpolator_data->fill(0);

    extractor_data_calc.fill_buffer(pre_req_extract_data, particle_sub_group, i,
                                    i + 1);

    particle_loop(
        "Interpolator loop", particle_sub_group,
        [=](auto particle_index, auto extracted_dat, auto req_int_props,
            auto req_real_props, auto kernel, auto interpolated_dat) {
          auto current_count = particle_index.get_loop_linear_index();
          std::array<REAL, ndim> transfer_arr;
          transfer_arr.fill(0.0);
          for (int idim = 0; idim < ndim; idim++) {
            transfer_arr[idim] = extracted_dat.at(current_count, idim);
          }

          std::array<REAL, ndim> output_dat =
              interpolator_data_on_device.calc_data(
                  transfer_arr, particle_index, req_int_props, req_real_props,
                  kernel);

          for (int idim = 0; idim < ndim; idim++) {
            interpolated_dat.at(current_count, idim) = output_dat[idim];
          }
        },
        Access::read(ParticleLoopIndex{}), Access::read(pre_req_extract_data),
        Access::write(sym_vector<INT>(particle_sub_group, req_int_props_)),
        Access::read(sym_vector<REAL>(particle_sub_group, req_real_props_)),
        Access::read(interpolator_data.get_rng_kernel()),
        Access::write(pre_req_interpolator_data))
        ->execute(i, i + 1);

    auto results_dat = pre_req_interpolator_data->get();

    for (int ipart = 0; ipart < pre_req_interpolator_data->index.shape[0];
         ipart++) {
      auto dim_size = pre_req_interpolator_data->index.shape[1];

      // Slightly different assertion test since function for the grid values is
      // log10(x), and grid spacing is sufficiently wide that linear
      // interpolation will not give an approximation that's compatible with
      // EXPECT_DOUBLE_EQ.
      EXPECT_NEAR(results_dat[(ipart * dim_size)], expected_interp_value, 1e-2);
    }
  }
}

TEST(InterpolationTest, REACTION_DATA_2D) {
  // Interpolation points
  REAL prop_interp_0 = 6.4e18;
  REAL prop_interp_1 = 1.9e3;
  REAL expected_interp_value = prop_interp_0 * prop_interp_1;
  static constexpr int ndim = 2;

  auto particle_group = create_test_particle_group(1e5);

  auto sycl_target = particle_group->sycl_target;

  auto npart = particle_group->get_npart_local();

  // This is a bit of a workaround for the fact that currently I can't use
  // PipelineData to pass a ConcatenatorData object that contains data from
  // multiple 1D ExtractorData objects and pass that to an InterpolateData
  // object.
  particle_group->add_particle_dat(Sym<REAL>("COMBINED_PROP"), ndim);

  particle_loop(
      particle_group,
      [=](auto prop) {
        prop.at(0) = prop_interp_0;
        prop.at(1) = prop_interp_1;
      },
      Access::write(Sym<REAL>("COMBINED_PROP")))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  // Setup the mock ADAS data values.
  auto ADAS_values = coefficient_values_2D();
  auto dims_vec = ADAS_values.get_dims_vec();
  auto ranges_vec = ADAS_values.get_ranges_flat_vec();
  auto grid = ADAS_values.get_coeffs_vec();

  // The input and output ndims being 2  for the InterpolateData (for this
  // specific unit test) is because PipelineData only presents the output ndims
  // of the last object in the pipeline when DataCalculator queries the
  // ndims of the passed PipelineData object. This is fine if either all objects
  // in the pipeline have the same output ndims OR if the last output ndims is
  // greater than the output ndims of any preceeding object in the pipeline.
  // For the current implementation of InterpolateData, I would
  // prefer input ndim be equal to the dimensionality of the grid.
  // and output ndim being 1 but this works fine for testing. (I'm just storing
  // the result in the 0th index of the array returned by
  // InterpolateData.calc_data and test against that).
  auto extractor_data = ExtractorData<ndim>(Sym<REAL>("COMBINED_PROP"));
  auto interpolator_data =
      InterpolateData<ndim>(dims_vec, ranges_vec, grid, sycl_target);
  auto interpolator_data_on_device = interpolator_data.get_on_device_obj();

  auto req_int_props_ = interpolator_data.get_required_int_sym_vector();
  auto req_real_props_ = interpolator_data.get_required_real_sym_vector();

  auto extractor_data_calc =
      DataCalculator<decltype(extractor_data)>(extractor_data);

  auto pre_req_extract_data = std::make_shared<NDLocalArray<REAL, 2>>(
      sycl_target, 0, extractor_data_calc.get_data_size());
  pre_req_extract_data->fill(0);

  auto pre_req_interpolator_data = std::make_shared<NDLocalArray<REAL, 2>>(
      sycl_target, 0, interpolator_data.get_dim());
  pre_req_interpolator_data->fill(0);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = pre_req_extract_data->index.shape;
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    pre_req_extract_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape[1]);
    pre_req_extract_data->fill(0);

    shape = pre_req_interpolator_data->index.shape;
    pre_req_interpolator_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape[1]);
    pre_req_interpolator_data->fill(0);

    extractor_data_calc.fill_buffer(pre_req_extract_data, particle_sub_group, i,
                                    i + 1);

    particle_loop(
        "Interpolator loop", particle_sub_group,
        [=](auto particle_index, auto extracted_dat, auto req_int_props,
            auto req_real_props, auto kernel, auto interpolated_dat) {
          auto current_count = particle_index.get_loop_linear_index();
          std::array<REAL, ndim> transfer_arr;
          transfer_arr.fill(0.0);
          for (int idim = 0; idim < ndim; idim++) {
            transfer_arr[idim] = extracted_dat.at(current_count, idim);
          }

          std::array<REAL, ndim> output_dat =
              interpolator_data_on_device.calc_data(
                  transfer_arr, particle_index, req_int_props, req_real_props,
                  kernel);

          for (int idim = 0; idim < ndim; idim++) {
            interpolated_dat.at(current_count, idim) = output_dat[idim];
          }
        },
        Access::read(ParticleLoopIndex{}), Access::read(pre_req_extract_data),
        Access::write(sym_vector<INT>(particle_sub_group, req_int_props_)),
        Access::read(sym_vector<REAL>(particle_sub_group, req_real_props_)),
        Access::read(interpolator_data.get_rng_kernel()),
        Access::write(pre_req_interpolator_data))
        ->execute(i, i + 1);

    auto results_dat = pre_req_interpolator_data->get();

    for (int ipart = 0; ipart < pre_req_interpolator_data->index.shape[0];
         ipart++) {
      auto dim_size = pre_req_interpolator_data->index.shape[1];

      EXPECT_DOUBLE_EQ(results_dat[(ipart * dim_size)], expected_interp_value);
    }
  }
}

// This is a prototype test for how I expect the pipelining to work for
// interpolation but I haven't gotten it to work yet.
#if 0
TEST(InterpolationTest, REACTION_DATA_2D_PIPELINE) {
  // Interpolation points
  REAL prop_interp_0 = 6.4e18;
  REAL prop_interp_1 = 1.9e3;
  REAL expected_interp_value = prop_interp_0 * prop_interp_1;
  static constexpr int ndim = 2;

  auto particle_group = create_test_particle_group(1e5);

  auto sycl_target = particle_group->sycl_target;

  auto npart = particle_group->get_npart_local();

  // This is a bit of a workaround for the fact that currently I can't use
  // PipelineData to pass a ConcatenatorData object that contains data from
  // multiple 1D ExtractorData objects and pass that to an InterpolateData
  // object.
  particle_group->add_particle_dat(Sym<REAL>("PROP1"), 1);
  particle_group->add_particle_dat(Sym<REAL>("PROP2"), 1);

  particle_loop(
      particle_group,
      [=](auto prop1, auto prop2) {
        prop1.at(0) = prop_interp_0;
        prop2.at(0) = prop_interp_1;
      },
      Access::write(Sym<REAL>("PROP1")),
      Access::write(Sym<REAL>("PROP2")))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  // Setup the mock ADAS data values.
  auto ADAS_values = coefficient_values_2D();
  auto dims_vec = ADAS_values.get_dims_vec();
  auto ranges_vec = ADAS_values.get_ranges_flat_vec();
  auto grid = ADAS_values.get_coeffs_vec();

  // The input and output ndims being 2  for the InterpolateData (for this
  // specific unit test) is because PipelineData only presents the output ndims
  // of the last object in the pipeline when DataCalculator queries the
  // ndims of the passed PipelineData object. This is fine if either all objects
  // in the pipeline have the same output ndims OR if the last output ndims is
  // greater than the output ndims of any preceeding object in the pipeline.
  // For the current implementation of InterpolateData, I would
  // prefer input ndim be equal to the dimensionality of the grid.
  // and output ndim being 1 but this works fine for testing. (I'm just storing
  // the result in the 0th index of the array returned by
  // InterpolateData.calc_data and test against that).
  auto concatenator = ConcatenatorData<ExtractorData<1>, ExtractorData<1>>(ExtractorData<1>(Sym<REAL>("PROP1")), ExtractorData<1>(Sym<REAL>("PROP2")));
  auto interpolator_data =
      InterpolateData<ndim>(dims_vec, ranges_vec, grid, sycl_target);

  auto pipeline = pipe(concatenator, interpolator_data);

  auto pipeline_data_calc = DataCalculator<decltype(pipeline)>(pipeline);

  auto pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
      sycl_target, 0, pipeline.get_dim());
  pre_req_data->fill(0);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = pre_req_data->index.shape;
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape[1]);
    pre_req_data->fill(0);

    pipeline_data_calc.fill_buffer(pre_req_data, particle_sub_group, i,
                                    i + 1);

    auto results_dat = pre_req_data->get();

    for (int ipart = 0; ipart < pre_req_data->index.shape[0];
         ipart++) {
      auto dim_size = pre_req_data->index.shape[1];

      printf("particle (%d) in cell (%d): %e\n", ipart, i, results_dat[(ipart * dim_size)]);
      // EXPECT_DOUBLE_EQ(results_dat[(ipart * dim_size)], expected_interp_value);
    }
  }
}
#endif

TEST(InterpolationTest, REACTION_DATA_3D) {
  // Interpolation points
  REAL prop_interp_0 = 6.4e18;
  REAL prop_interp_1 = 1.9e3;
  REAL prop_interp_2 = 2.3e4;
  REAL expected_interp_value = prop_interp_0 * prop_interp_1 * prop_interp_2;
  static constexpr int ndim = 3;

  auto particle_group = create_test_particle_group(1e5);

  auto sycl_target = particle_group->sycl_target;

  auto npart = particle_group->get_npart_local();

  // This is a bit of a workaround for the fact that currently I can't use
  // PipelineData to pass a ConcatenatorData object that contains data from
  // multiple 1D ExtractorData objects and pass that to an InterpolateData
  // object.
  particle_group->add_particle_dat(Sym<REAL>("COMBINED_PROP"), ndim);

  particle_loop(
      particle_group,
      [=](auto prop) {
        prop.at(0) = prop_interp_0;
        prop.at(1) = prop_interp_1;
        prop.at(2) = prop_interp_2;
      },
      Access::write(Sym<REAL>("COMBINED_PROP")))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  // Setup the mock ADAS data values.
  auto ADAS_values = coefficient_values_3D();
  auto dims_vec = ADAS_values.get_dims_vec();
  auto ranges_vec = ADAS_values.get_ranges_flat_vec();
  auto grid = ADAS_values.get_coeffs_vec();

  // The input and output ndims being 2  for the InterpolateData (for this
  // specific unit test) is because PipelineData only presents the output ndims
  // of the last object in the pipeline when DataCalculator queries the
  // ndims of the passed PipelineData object. This is fine if either all objects
  // in the pipeline have the same output ndims OR if the last output ndims is
  // greater than the output ndims of any preceeding object in the pipeline.
  // For the current implementation of InterpolateData, I would
  // prefer input ndim be equal to the dimensionality of the grid.
  // and output ndim being 1 but this works fine for testing. (I'm just storing
  // the result in the 0th index of the array returned by
  // InterpolateData.calc_data and test against that).
  auto extractor_data = ExtractorData<ndim>(Sym<REAL>("COMBINED_PROP"));
  auto interpolator_data =
      InterpolateData<ndim>(dims_vec, ranges_vec, grid, sycl_target);
  auto interpolator_data_on_device = interpolator_data.get_on_device_obj();

  auto req_int_props_ = interpolator_data.get_required_int_sym_vector();
  auto req_real_props_ = interpolator_data.get_required_real_sym_vector();

  auto extractor_data_calc =
      DataCalculator<decltype(extractor_data)>(extractor_data);

  auto pre_req_extract_data = std::make_shared<NDLocalArray<REAL, 2>>(
      sycl_target, 0, extractor_data_calc.get_data_size());
  pre_req_extract_data->fill(0);

  auto pre_req_interpolator_data = std::make_shared<NDLocalArray<REAL, 2>>(
      sycl_target, 0, interpolator_data.get_dim());
  pre_req_interpolator_data->fill(0);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = pre_req_extract_data->index.shape;
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    pre_req_extract_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape[1]);
    pre_req_extract_data->fill(0);

    shape = pre_req_interpolator_data->index.shape;
    pre_req_interpolator_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape[1]);
    pre_req_interpolator_data->fill(0);

    extractor_data_calc.fill_buffer(pre_req_extract_data, particle_sub_group, i,
                                    i + 1);

    particle_loop(
        "Interpolator loop", particle_sub_group,
        [=](auto particle_index, auto extracted_dat, auto req_int_props,
            auto req_real_props, auto kernel, auto interpolated_dat) {
          auto current_count = particle_index.get_loop_linear_index();
          std::array<REAL, ndim> transfer_arr;
          transfer_arr.fill(0.0);
          for (int idim = 0; idim < ndim; idim++) {
            transfer_arr[idim] = extracted_dat.at(current_count, idim);
          }

          std::array<REAL, ndim> output_dat =
              interpolator_data_on_device.calc_data(
                  transfer_arr, particle_index, req_int_props, req_real_props,
                  kernel);

          for (int idim = 0; idim < ndim; idim++) {
            interpolated_dat.at(current_count, idim) = output_dat[idim];
          }
        },
        Access::read(ParticleLoopIndex{}), Access::read(pre_req_extract_data),
        Access::write(sym_vector<INT>(particle_sub_group, req_int_props_)),
        Access::read(sym_vector<REAL>(particle_sub_group, req_real_props_)),
        Access::read(interpolator_data.get_rng_kernel()),
        Access::write(pre_req_interpolator_data))
        ->execute(i, i + 1);

    auto results_dat = pre_req_interpolator_data->get();

    for (int ipart = 0; ipart < pre_req_interpolator_data->index.shape[0];
         ipart++) {
      auto dim_size = pre_req_interpolator_data->index.shape[1];

      EXPECT_DOUBLE_EQ(results_dat[(ipart * dim_size)], expected_interp_value);
    }
  }
}

TEST(InterpolationTest, REACTION_DATA_4D) {
  // Interpolation points
  REAL prop_interp_0 = 6.4e18;
  REAL prop_interp_1 = 1.9e3;
  REAL prop_interp_2 = 2.3e4;
  REAL prop_interp_3 = 3.2e7;
  REAL expected_interp_value =
      prop_interp_0 * prop_interp_1 * prop_interp_2 * prop_interp_3;
  static constexpr int ndim = 4;

  auto particle_group = create_test_particle_group(1e5);

  auto sycl_target = particle_group->sycl_target;

  auto npart = particle_group->get_npart_local();

  // This is a bit of a workaround for the fact that currently I can't use
  // PipelineData to pass a ConcatenatorData object that contains data from
  // multiple 1D ExtractorData objects and pass that to an InterpolateData
  // object.
  particle_group->add_particle_dat(Sym<REAL>("COMBINED_PROP"), ndim);

  particle_loop(
      particle_group,
      [=](auto prop) {
        prop.at(0) = prop_interp_0;
        prop.at(1) = prop_interp_1;
        prop.at(2) = prop_interp_2;
        prop.at(3) = prop_interp_3;
      },
      Access::write(Sym<REAL>("COMBINED_PROP")))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  // Setup the mock ADAS data values.
  auto ADAS_values = coefficient_values_4D();
  auto dims_vec = ADAS_values.get_dims_vec();
  auto ranges_vec = ADAS_values.get_ranges_flat_vec();
  auto grid = ADAS_values.get_coeffs_vec();

  // The input and output ndims being 2  for the InterpolateData (for this
  // specific unit test) is because PipelineData only presents the output ndims
  // of the last object in the pipeline when DataCalculator queries the
  // ndims of the passed PipelineData object. This is fine if either all objects
  // in the pipeline have the same output ndims OR if the last output ndims is
  // greater than the output ndims of any preceeding object in the pipeline.
  // For the current implementation of InterpolateData, I would
  // prefer input ndim be equal to the dimensionality of the grid.
  // and output ndim being 1 but this works fine for testing. (I'm just storing
  // the result in the 0th index of the array returned by
  // InterpolateData.calc_data and test against that).
  auto extractor_data = ExtractorData<ndim>(Sym<REAL>("COMBINED_PROP"));
  auto interpolator_data =
      InterpolateData<ndim>(dims_vec, ranges_vec, grid, sycl_target);
  auto interpolator_data_on_device = interpolator_data.get_on_device_obj();

  auto req_int_props_ = interpolator_data.get_required_int_sym_vector();
  auto req_real_props_ = interpolator_data.get_required_real_sym_vector();

  auto extractor_data_calc =
      DataCalculator<decltype(extractor_data)>(extractor_data);

  auto pre_req_extract_data = std::make_shared<NDLocalArray<REAL, 2>>(
      sycl_target, 0, extractor_data_calc.get_data_size());
  pre_req_extract_data->fill(0);

  auto pre_req_interpolator_data = std::make_shared<NDLocalArray<REAL, 2>>(
      sycl_target, 0, interpolator_data.get_dim());
  pre_req_interpolator_data->fill(0);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = pre_req_extract_data->index.shape;
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    pre_req_extract_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape[1]);
    pre_req_extract_data->fill(0);

    shape = pre_req_interpolator_data->index.shape;
    pre_req_interpolator_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape[1]);
    pre_req_interpolator_data->fill(0);

    extractor_data_calc.fill_buffer(pre_req_extract_data, particle_sub_group, i,
                                    i + 1);

    particle_loop(
        "Interpolator loop", particle_sub_group,
        [=](auto particle_index, auto extracted_dat, auto req_int_props,
            auto req_real_props, auto kernel, auto interpolated_dat) {
          auto current_count = particle_index.get_loop_linear_index();
          std::array<REAL, ndim> transfer_arr;
          transfer_arr.fill(0.0);
          for (int idim = 0; idim < ndim; idim++) {
            transfer_arr[idim] = extracted_dat.at(current_count, idim);
          }

          std::array<REAL, ndim> output_dat =
              interpolator_data_on_device.calc_data(
                  transfer_arr, particle_index, req_int_props, req_real_props,
                  kernel);

          for (int idim = 0; idim < ndim; idim++) {
            interpolated_dat.at(current_count, idim) = output_dat[idim];
          }
        },
        Access::read(ParticleLoopIndex{}), Access::read(pre_req_extract_data),
        Access::write(sym_vector<INT>(particle_sub_group, req_int_props_)),
        Access::read(sym_vector<REAL>(particle_sub_group, req_real_props_)),
        Access::read(interpolator_data.get_rng_kernel()),
        Access::write(pre_req_interpolator_data))
        ->execute(i, i + 1);

    auto results_dat = pre_req_interpolator_data->get();

    for (int ipart = 0; ipart < pre_req_interpolator_data->index.shape[0];
         ipart++) {
      auto dim_size = pre_req_interpolator_data->index.shape[1];

      auto relative_error =
          std::abs(results_dat[(ipart * dim_size)] - expected_interp_value) /
          std::abs(expected_interp_value);

      // Linear interpolation seems to be running out of steam with regards to
      // precision hence the check is to see if the calculated value is within
      // 1e-6 for relative error.
      EXPECT_NEAR(relative_error, 0.0, 1e-6);
    }
  }
}

TEST(InterpolationTest, REACTION_DATA_5D) {
  // Interpolation points
  REAL prop_interp_0 = 6.4e18;
  REAL prop_interp_1 = 1.9e3;
  REAL prop_interp_2 = 2.3e4;
  REAL prop_interp_3 = 3.2e7;
  REAL prop_interp_4 = 2.07;
  REAL expected_interp_value = prop_interp_0 * prop_interp_1 * prop_interp_2 *
                               prop_interp_3 * prop_interp_4;
  static constexpr int ndim = 5;

  // Had to reduce this down from 1e5 due to some memory limitation on GPU runs.
  auto particle_group = create_test_particle_group(1e4);

  auto sycl_target = particle_group->sycl_target;

  auto npart = particle_group->get_npart_local();

  // This is a bit of a workaround for the fact that currently I can't use
  // PipelineData to pass a ConcatenatorData object that contains data from
  // multiple 1D ExtractorData objects and pass that to an InterpolateData
  // object.
  particle_group->add_particle_dat(Sym<REAL>("COMBINED_PROP"), ndim);

  particle_loop(
      particle_group,
      [=](auto prop) {
        prop.at(0) = prop_interp_0;
        prop.at(1) = prop_interp_1;
        prop.at(2) = prop_interp_2;
        prop.at(3) = prop_interp_3;
        prop.at(4) = prop_interp_4;
      },
      Access::write(Sym<REAL>("COMBINED_PROP")))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  // Setup the mock ADAS data values.
  auto ADAS_values = coefficient_values_5D();
  auto dims_vec = ADAS_values.get_dims_vec();
  auto ranges_vec = ADAS_values.get_ranges_flat_vec();
  auto grid = ADAS_values.get_coeffs_vec();

  // The input and output ndims being 2  for the InterpolateData (for this
  // specific unit test) is because PipelineData only presents the output ndims
  // of the last object in the pipeline when DataCalculator queries the
  // ndims of the passed PipelineData object. This is fine if either all objects
  // in the pipeline have the same output ndims OR if the last output ndims is
  // greater than the output ndims of any preceeding object in the pipeline.
  // For the current implementation of InterpolateData, I would
  // prefer input ndim be equal to the dimensionality of the grid.
  // and output ndim being 1 but this works fine for testing. (I'm just storing
  // the result in the 0th index of the array returned by
  // InterpolateData.calc_data and test against that).
  auto extractor_data = ExtractorData<ndim>(Sym<REAL>("COMBINED_PROP"));
  auto interpolator_data =
      InterpolateData<ndim>(dims_vec, ranges_vec, grid, sycl_target);
  auto interpolator_data_on_device = interpolator_data.get_on_device_obj();

  auto req_int_props_ = interpolator_data.get_required_int_sym_vector();
  auto req_real_props_ = interpolator_data.get_required_real_sym_vector();

  auto extractor_data_calc =
      DataCalculator<decltype(extractor_data)>(extractor_data);

  auto pre_req_extract_data = std::make_shared<NDLocalArray<REAL, 2>>(
      sycl_target, 0, extractor_data_calc.get_data_size());
  pre_req_extract_data->fill(0);

  auto pre_req_interpolator_data = std::make_shared<NDLocalArray<REAL, 2>>(
      sycl_target, 0, interpolator_data.get_dim());
  pre_req_interpolator_data->fill(0);

  const int cell_count = particle_group->domain->mesh->get_cell_count();

  for (int i = 0; i < cell_count; i++) {
    auto shape = pre_req_extract_data->index.shape;
    auto n_part_cell = particle_sub_group->get_npart_cell(i);
    size_t buffer_size = n_part_cell;
    pre_req_extract_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape[1]);
    pre_req_extract_data->fill(0);

    shape = pre_req_interpolator_data->index.shape;
    pre_req_interpolator_data = std::make_shared<NDLocalArray<REAL, 2>>(
        particle_group->sycl_target, buffer_size, shape[1]);
    pre_req_interpolator_data->fill(0);

    extractor_data_calc.fill_buffer(pre_req_extract_data, particle_sub_group, i,
                                    i + 1);

    particle_loop(
        "Interpolator loop", particle_sub_group,
        [=](auto particle_index, auto extracted_dat, auto req_int_props,
            auto req_real_props, auto kernel, auto interpolated_dat) {
          auto current_count = particle_index.get_loop_linear_index();
          std::array<REAL, ndim> transfer_arr;
          transfer_arr.fill(0.0);
          for (int idim = 0; idim < ndim; idim++) {
            transfer_arr[idim] = extracted_dat.at(current_count, idim);
          }

          std::array<REAL, ndim> output_dat =
              interpolator_data_on_device.calc_data(
                  transfer_arr, particle_index, req_int_props, req_real_props,
                  kernel);

          for (int idim = 0; idim < ndim; idim++) {
            interpolated_dat.at(current_count, idim) = output_dat[idim];
          }
        },
        Access::read(ParticleLoopIndex{}), Access::read(pre_req_extract_data),
        Access::write(sym_vector<INT>(particle_sub_group, req_int_props_)),
        Access::read(sym_vector<REAL>(particle_sub_group, req_real_props_)),
        Access::read(interpolator_data.get_rng_kernel()),
        Access::write(pre_req_interpolator_data))
        ->execute(i, i + 1);

    auto results_dat = pre_req_interpolator_data->get();

    for (int ipart = 0; ipart < pre_req_interpolator_data->index.shape[0];
         ipart++) {
      auto dim_size = pre_req_interpolator_data->index.shape[1];

      auto relative_error =
          std::abs(results_dat[(ipart * dim_size)] - expected_interp_value) /
          std::abs(expected_interp_value);

      // Linear interpolation seems to be running out of steam with regards to
      // precision hence the check is to see if the calculated value is within
      // 1e-6 for relative error.
      EXPECT_NEAR(relative_error, 0.0, 1e-6);
    }
  }
}
