#ifndef REACTIONS_MOCK_INTERPOLATION_DATA_H
#define REACTIONS_MOCK_INTERPOLATION_DATA_H
#include <neso_particles.hpp>
#include <reactions/reactions.hpp>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

struct abstract_coefficient_values {
  abstract_coefficient_values() = default;

  virtual ~abstract_coefficient_values() = default;

protected:
  std::vector<REAL> coeffs_vec;
  std::vector<REAL> ranges_flat_vec;
  std::vector<REAL> lower_bounds;
  std::vector<REAL> upper_bounds;
  std::vector<size_t> dims_vec;

public:
  const std::vector<REAL> &get_coeffs_vec() { return this->coeffs_vec; }

  const std::vector<REAL> &get_ranges_flat_vec() {
    return this->ranges_flat_vec;
  }

  const std::vector<size_t> &get_dims_vec() { return this->dims_vec; }

  const std::vector<REAL> &get_lower_bounds() { return this->lower_bounds; }

  const std::vector<REAL> &get_upper_bounds() { return this->upper_bounds; }
};

struct coefficient_values_1D_linear : abstract_coefficient_values {
private:
  static constexpr int ndim = 1;
  static constexpr size_t dim0 = 8;

  // Generated with python: numpy.linspace(1.0e18, 8.0e18, 8)
  std::vector<REAL> dim0_range = {1.0e+18, 2.0e+18, 3.0e+18, 4.0e+18,
                                  5.0e+18, 6.0e+18, 7.0e+18, 8.0e+18};

public:
  coefficient_values_1D_linear() : abstract_coefficient_values() {
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

  REAL grid_func(const REAL &dim0_val) const { return (2 * dim0_val); }
};

struct coefficient_values_1D : abstract_coefficient_values {
private:
  static constexpr int ndim = 1;
  static constexpr size_t dim0 = 8;

  // Generated with python: numpy.linspace(1.0e18, 8.0e18, 8)
  std::vector<REAL> dim0_range = {1.0e+18, 2.0e+18, 3.0e+18, 4.0e+18,
                                  5.0e+18, 6.0e+18, 7.0e+18, 8.0e+18};

public:
  coefficient_values_1D() : abstract_coefficient_values() {
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

  REAL grid_func(const REAL &val) const { return std::log10(val); }
};

struct coefficient_values_2D : abstract_coefficient_values {
private:
  static constexpr int ndim = 2;
  static const size_t dim0 = 8;
  static const size_t dim1 = 10;

  // Generated with python: numpy.linspace(1.0e18, 8.0e18, 8)
  std::vector<REAL> dim0_range = {1.0e+18, 2.0e+18, 3.0e+18, 4.0e+18,
                                  5.0e+18, 6.0e+18, 7.0e+18, 8.0e+18};
  // Generated with python: numpy.logspace(1, 5, 10)
  std::vector<REAL> dim1_range = {
      1.00000000e+01, 2.78255940e+01, 7.74263683e+01, 2.15443469e+02,
      5.99484250e+02, 1.66810054e+03, 4.64158883e+03, 1.29154967e+04,
      3.59381366e+04, 1.00000000e+05};

public:
  coefficient_values_2D() : abstract_coefficient_values() {
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

  REAL grid_func(const REAL &dim0_val, const REAL &dim1_val) const {
    return (dim0_val * dim1_val);
  }
};

struct coefficient_values_3D : abstract_coefficient_values {
private:
  static constexpr int ndim = 3;
  static constexpr size_t dim0 = 8;
  static constexpr size_t dim1 = 10;
  static constexpr size_t dim2 = 15;

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

public:
  coefficient_values_3D() : abstract_coefficient_values() {
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

  REAL grid_func(const REAL &dim0_val, const REAL &dim1_val,
                 const REAL &dim2_val) const {
    return (dim0_val * dim1_val * dim2_val);
  }
};

struct coefficient_values_4D : abstract_coefficient_values {
private:
  static constexpr int ndim = 4;
  static constexpr size_t dim0 = 8;
  static constexpr size_t dim1 = 10;
  static constexpr size_t dim2 = 15;
  static constexpr size_t dim3 = 23;

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

public:
  coefficient_values_4D() : abstract_coefficient_values() {
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

  REAL grid_func(const REAL &dim0_val, const REAL &dim1_val,
                 const REAL &dim2_val, const REAL &dim3_val) const {
    return (dim0_val * dim1_val * dim2_val * dim3_val);
  }
};

struct coefficient_values_5D : abstract_coefficient_values {
private:
  static constexpr int ndim = 5;
  static constexpr size_t dim0 = 8;
  static constexpr size_t dim1 = 10;
  static constexpr size_t dim2 = 15;
  static constexpr size_t dim3 = 23;
  static constexpr size_t dim4 = 35;

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

public:
  coefficient_values_5D() : abstract_coefficient_values() {
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

  REAL grid_func(const REAL &dim0_val, const REAL &dim1_val,
                 const REAL &dim2_val, const REAL &dim3_val,
                 const REAL &dim4_val) const {
    return (dim0_val * dim1_val * dim2_val * dim3_val * dim4_val);
  }
};

#endif
