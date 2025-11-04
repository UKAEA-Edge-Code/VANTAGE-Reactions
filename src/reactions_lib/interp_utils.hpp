#ifndef REACTIONS_INTERP_UTILS_H
#define REACTIONS_INTERP_UTILS_H
#include <cassert>
#include <cmath>
#include <cstddef>
#include <neso_particles.hpp>
#include <vector>

#define binary_extract(i, j) ((i >> j) & 1)

using namespace NESO::Particles;
namespace VANTAGE::Reactions::interp_utils {

inline size_t coeff_index(const std::vector<size_t> &indices,
                          const std::vector<size_t> &dims_vec) {
  const size_t ndim = indices.size();

  size_t index = indices[0];

  size_t dim_mult;
  for (size_t i = 1; i < ndim; i++) {
    dim_mult = 1;
    for (int j = int(i - 1); j >= 0; j--) {
      dim_mult *= dims_vec[j];
    }
    index += dim_mult * indices[i];
  }

  return index;
}

inline size_t coeff_index_on_device(
    Access::LocalMemoryInterlaced::Write<unsigned long> indices,
    size_t *dims_vec, const int &ndim) {
  size_t index = indices.at(0);

  size_t dim_mult;
  for (size_t i = 1; i < ndim; i++) {
    dim_mult = 1;
    for (int j = int(i - 1); j >= 0; j--) {
      dim_mult *= dims_vec[j];
    }
    index += dim_mult * indices.at(i);
  }

  return index;
}

inline size_t range_index(const size_t &sub_index, const size_t &dim_index,
                          const std::vector<size_t> &dims_vec) {

  size_t index = sub_index;

  for (size_t i = 0; i < dim_index; i++) {
    index += dims_vec[i];
  }

  return index;
}

inline size_t range_index_on_device(const size_t &sub_index,
                                    const size_t &dim_index, size_t *dims_vec) {
  size_t index = sub_index;

  for (size_t i = 0; i < dim_index; i++) {
    index += dims_vec[i];
  }

  return index;
}

inline std::size_t calc_closest_point_index(const REAL &x_interp,
                                            REAL *dim_range,
                                            const std::size_t &dim_size) {
  std::size_t L = 0;
  std::size_t R = dim_size - 1;
  std::size_t m;

  while ((R - L) > 1) {
    m = ((L + R) / 2);
    if (dim_range[m] < x_interp) {
      L = m;
    } else if (dim_range[m] > x_interp) {
      R = m;
    } else {
      // for exact matches
      return m;
    }
  }

  return R == (dim_size - 1) ? R : L;
}

inline REAL linear_interp(const REAL &x_interp, const REAL &x0, const REAL &x1,
                          const REAL &f0, const REAL &f1) {
  REAL dfdx = (f1 - f0) / (x1 - x0);
  REAL c = f0 - (dfdx * x0);

  REAL f_interp = (dfdx * x_interp) + c;
  return f_interp;
}

inline std::vector<int> construct_initial_hypercube(const int &ndim) {
  int total_num = 1 << ndim;
  std::vector<int> points(total_num);

  for (int i = 0; i < total_num; i++) {
    points[i] = (i ^ (i >> 1));
  }

  return points;
}

inline std::vector<double> initial_func_eval(
    const std::vector<double> &func_grid, const std::vector<int> &vertices,
    const std::vector<size_t> &origin_indices,
    const std::vector<size_t> &dims_vec, const size_t &dim_index) {
  int num_points = (1 << (dim_index + 1));
  size_t ndim = dim_index + 1;

  std::vector<double> vertex_func_evals(num_points);
  std::vector<size_t> eval_point(ndim);

  for (int i = 0; i < num_points; i++) {
    for (int j = size_t(dim_index); j >= 0; j--) {
      eval_point[j] = origin_indices[j] + binary_extract(vertices[i], j);
    }

    vertex_func_evals[i] = func_grid[coeff_index(eval_point, dims_vec)];
  }

  return vertex_func_evals;
}

inline void initial_func_eval_on_device(
    Access::LocalMemoryInterlaced::Write<REAL> vertex_func_evals,
    Access::LocalMemoryInterlaced::Write<size_t> vertex_coord, REAL *func_grid,
    int *hypercube_vertices,
    Access::LocalMemoryInterlaced::Write<size_t> origin_indices,
    size_t *dims_vec, const int &ndim) {
  int num_points = (1 << ndim);

  for (int i = 0; i < num_points; i++) {
    for (int j = size_t(ndim) - 1; j >= 0; j--) {
      vertex_coord.at(j) =
          origin_indices.at(j) + binary_extract(hypercube_vertices[i], j);
    }
    vertex_func_evals.at(i) =
        func_grid[coeff_index_on_device(vertex_coord, dims_vec, ndim)];
  }
}

inline std::tuple<std::vector<int>, std::vector<double>>
contract_hypercube(const std::vector<double> &interp_points, int dim_index,
                   const std::vector<int> &points,
                   const std::vector<size_t> origin_indices,
                   const std::vector<double> &vertex_func_evals,
                   const std::vector<double> &ranges_vec,
                   const std::vector<size_t> &dims_vec) {
  int num_points = (1 << (dim_index + 1));
  size_t ndim = dim_index + 1;

  std::vector<size_t> varying_dim(num_points);
  std::vector<size_t> eval_point(ndim);

  for (int i = 0; i < num_points; i++) {
    for (int j = size_t(dim_index); j >= 0; j--) {
      eval_point[j] = origin_indices[j] + binary_extract(points[i], j);
      // printf("%ld", eval_point[j]);
    }

    varying_dim[i] = eval_point[dim_index];
    // printf("\t%e\t%ld", vertex_func_evals[i], varying_dim[i]);
    // printf("\n");
  }

  std::vector<double> output_evals(std::size_t(1 << dim_index));
  std::vector<int> output_points(std::size_t(1 << dim_index));

  size_t vertex_0, vertex_1;

  size_t vertex_out;

  for (int i = 0; i < (1 << dim_index); i++) {
    // printf("%d\t", i);
    vertex_0 = varying_dim[i];
    vertex_1 = (*(varying_dim.end() - (i + 1)));

    double range_val_0 = ranges_vec[range_index(vertex_0, 0, dims_vec)];
    double range_val_1 = ranges_vec[range_index(vertex_1, 0, dims_vec)];

    double eval_point_0 = vertex_func_evals[i];
    double eval_point_1 = (*(vertex_func_evals.end() - (i + 1)));
    // printf("%ld,%e,%e\t%ld,%e,%e\n", vertex_0, range_val_0, eval_point_0,
    //        vertex_1, range_val_1, eval_point_1);

    output_evals[i] = linear_interp(interp_points[0], range_val_0, range_val_1,
                                    eval_point_0, eval_point_1);

    vertex_out = points[i];

    output_points[i] = int(vertex_out);
  }

  return std::make_tuple(output_points, output_evals);
}

inline void contract_hypercube_on_device(
    const INT &particle_count,
    Access::LocalMemoryInterlaced::Write<REAL> interp_points,
    const int &dim_index,
    Access::LocalMemoryInterlaced::Write<int> input_vertices,
    Access::LocalMemoryInterlaced::Write<size_t> origin_indices,
    Access::LocalMemoryInterlaced::Write<REAL> vertex_func_evals,
    REAL *ranges_vec, size_t *dims_vec,
    Access::LocalMemoryInterlaced::Write<int> output_vertices,
    Access::LocalMemoryInterlaced::Write<REAL> output_evals,
    Access::LocalMemoryInterlaced::Write<size_t> varying_dim,
    Access::LocalMemoryInterlaced::Write<size_t> eval_point) {
  int ndim = dim_index + 1;
  int num_points = (1 << ndim);
  int num_out_points = (1 << dim_index);

  for (int i = 0; i < num_points; i++) {
    for (int j = dim_index; j >= 0; j--) {
      eval_point.at(j) =
          origin_indices.at(j) + binary_extract(input_vertices.at(i), j);
    }

    varying_dim.at(i) = eval_point.at(dim_index);
  }

  size_t vertex_0, vertex_1;
  REAL range_val_0, range_val_1, eval_point_0, eval_point_1;

  for (int i = 0; i < num_out_points; i++) {
    vertex_0 = varying_dim.at(i);
    vertex_1 = varying_dim.at(num_points - (i + 1));

    range_val_0 =
        ranges_vec[range_index_on_device(vertex_0, dim_index, dims_vec)];
    range_val_1 =
        ranges_vec[range_index_on_device(vertex_1, dim_index, dims_vec)];

    eval_point_0 = vertex_func_evals.at(i);
    eval_point_1 = vertex_func_evals.at(num_points - (i + 1));

    output_evals.at(i) = linear_interp(interp_points.at(dim_index), range_val_0,
                                       range_val_1, eval_point_0, eval_point_1);

    // if (particle_count == 0) {
    //   printf("interp_point: %e\n", interp_points.at(0));
    //   printf("vertex_0: %ld\n", vertex_0);
    //   printf("vertex_1: %ld\n", vertex_1);
    //   printf("range_val0: %e\n", range_val_0);
    //   printf("range_val1: %e\n", range_val_1);
    //   printf("eval_point_0: %e\n", eval_point_0);
    //   printf("eval_point_1: %e\n", eval_point_1);
    //   printf("output_eval_%d: %e\n", i, output_evals.at(i));
    //   printf("\n");
    // }

    output_vertices.at(i) = input_vertices.at(i);
  }
}

} // namespace VANTAGE::Reactions::interp_utils
#endif
