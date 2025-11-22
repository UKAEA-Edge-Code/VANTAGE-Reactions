#ifndef REACTIONS_INTERP_UTILS_H
#define REACTIONS_INTERP_UTILS_H
#include <cassert>
#include <cmath>
#include <cstddef>
#include <neso_particles.hpp>
#include <neso_particles/containers/local_memory_interlaced.hpp>
#include <vector>

#define binary_extract(i, j) ((i >> j) & 1)

using namespace NESO::Particles;
namespace VANTAGE::Reactions::interp_utils {

inline std::size_t
coeff_index_on_device(Access::LocalMemoryInterlaced::Write<std::size_t> indices,
                      const int &point_index, std::size_t *dims_vec,
                      const int &ndim) {
  std::size_t index = indices.at(0);

  std::size_t dim_mult;
  for (int i = 1; i < ndim; i++) {
    dim_mult = 1;
    for (int j = int(i - 1); j >= 0; j--) {
      dim_mult *= dims_vec[j];
    }
    index += dim_mult * indices.at(i);
  }

  return index;
}

inline INT coeff_index_on_device(INT *indices, const int &point_index,
                                 size_t *dims_vec, const int &ndim) {
  INT index = indices[0];

  std::size_t dim_mult;
  for (int i = 1; i < ndim; i++) {
    dim_mult = 1;
    for (int j = int(i - 1); j >= 0; j--) {
      dim_mult *= dims_vec[j];
    }
    index += dim_mult * indices[i];
  }

  return index;
}

inline std::size_t range_index_on_device(const std::size_t &sub_index,
                                         const std::size_t &dim_index,
                                         std::size_t *dims_vec) {
  std::size_t index = sub_index;

  for (std::size_t i = 0; i < dim_index; i++) {
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

inline std::vector<INT> construct_initial_hypercube(const INT &ndim) {
  int total_num = 1 << ndim;
  std::vector<INT> points(total_num);

  for (int i = 0; i < total_num; i++) {
    points[i] = (i ^ (i >> 1));
  }

  return points;
}

inline std::vector<int> construct_initial_hypercube(const int &ndim) {
  int total_num = 1 << ndim;
  std::vector<int> points(total_num);

  for (int i = 0; i < total_num; i++) {
    points[i] = (i ^ (i >> 1));
  }

  return points;
}

inline void initial_func_eval_on_device(
    Access::LocalMemoryInterlaced::Write<REAL> vertex_func_evals,
    Access::LocalMemoryInterlaced::Write<std::size_t> vertex_coord,
    REAL *func_grid, int *hypercube_vertices,
    Access::LocalMemoryInterlaced::Write<std::size_t> origin_indices,
    std::size_t *dims_vec, const int &ndim) {
  int num_points = (1 << ndim);
  int point_index = 0;
  int vertex_index = 0;

  for (int i = 0; i < (num_points * ndim); i++) {
    point_index = i / ndim;
    vertex_index = (ndim - 1) - (i % ndim);

    vertex_coord.at(vertex_index) =
        origin_indices.at(vertex_index) +
        binary_extract(hypercube_vertices[point_index], vertex_index);

    if (i % ndim) {
      vertex_func_evals.at(point_index) = func_grid[coeff_index_on_device(
          vertex_coord, point_index, dims_vec, ndim)];
    }
  }
}

inline void initial_func_eval_on_device(REAL *vertex_func_evals,
                                        INT *vertex_coord, REAL *func_grid,
                                        INT *hypercube_vertices,
                                        INT *origin_indices, size_t *dims_vec,
                                        const int &ndim,
                                        const int &num_points) {
  int point_index = 0;
  int vertex_index = 0;

  for (int i = 0; i < (num_points * ndim); i++) {
    point_index = i / ndim;
    vertex_index = (ndim - 1) - (i % ndim);

    vertex_coord[vertex_index] =
        origin_indices[vertex_index] +
        binary_extract(hypercube_vertices[point_index], vertex_index);

    if (i % ndim) {
      vertex_func_evals[point_index] = func_grid[coeff_index_on_device(
          vertex_coord, point_index, dims_vec, ndim)];
    }
  }
}

inline void contract_hypercube_on_device(
    const INT &particle_count,
    Access::LocalMemoryInterlaced::Write<REAL> interp_points,
    const int &dim_index,
    Access::LocalMemoryInterlaced::Write<int> input_vertices,
    Access::LocalMemoryInterlaced::Write<std::size_t> origin_indices,
    Access::LocalMemoryInterlaced::Write<REAL> vertex_func_evals,
    REAL *ranges_vec, std::size_t *dims_vec,
    Access::LocalMemoryInterlaced::Write<int> output_vertices,
    Access::LocalMemoryInterlaced::Write<REAL> output_evals,
    Access::LocalMemoryInterlaced::Write<std::size_t> varying_dim,
    Access::LocalMemoryInterlaced::Write<std::size_t> vertex_coord) {
  int ndim = dim_index + 1;
  int num_points = (1 << ndim);
  int num_out_points = (1 << dim_index);

  int point_index = 0;
  int eval_index = 0;

  for (int i = 0; i < (num_points * ndim); i++) {
    point_index = i / ndim;
    eval_index = (ndim - 1) - (i % ndim);

    vertex_coord.at(eval_index) =
        origin_indices.at(eval_index) +
        binary_extract(input_vertices.at(point_index), eval_index);

    if ((ndim > 1) ? (i % ndim) : 1) {
      varying_dim.at(point_index) = vertex_coord.at(dim_index);
    }
  }

  std::size_t vertex_0, vertex_1;
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

    output_vertices.at(i) = input_vertices.at(i);

    // if (particle_count == 0) {
    //   printf("dim_index: %d\n", dim_index);
    //   printf("interp_point: %e\n", interp_points.at(0));
    //   printf("vertex_0: %ld\n", vertex_0);
    //   printf("vertex_1: %ld\n", vertex_1);
    //   printf("range_val0: %e\n", range_val_0);
    //   printf("range_val1: %e\n", range_val_1);
    //   printf("eval_point_0: %e\n", eval_point_0);
    //   printf("eval_point_1: %e\n", eval_point_1);
    //   printf("output_eval_%d: %e\n", i, output_evals.at(i));
    //   printf("output_vertices: %d\n", output_vertices.at(i));
    //   printf("\n");
    // }
  }
}

inline void contract_hypercube_on_device(
    const INT &particle_count, const REAL *interp_points, const int &dim_index,
    INT *input_vertices, INT *origin_indices, REAL *vertex_func_evals,
    REAL *ranges_vec, std::size_t *dims_vec, INT *output_vertices,
    REAL *output_evals, INT *varying_dim, INT *vertex_coord) {
  int ndim = dim_index + 1;
  int num_points = (1 << ndim);
  int num_out_points = (1 << dim_index);

  int point_index = 0;
  int eval_index = 0;

  for (int i = 0; i < (num_points * ndim); i++) {
    point_index = i / ndim;
    eval_index = (ndim - 1) - (i % ndim);

    vertex_coord[eval_index] =
        origin_indices[eval_index] +
        binary_extract(input_vertices[point_index], eval_index);

    if ((ndim > 1) ? (i % ndim) : 1) {
      varying_dim[point_index] = vertex_coord[dim_index];
    }
  }

  INT vertex_0, vertex_1;
  REAL range_val_0, range_val_1, eval_point_0, eval_point_1;

  for (int i = 0; i < num_out_points; i++) {
    vertex_0 = varying_dim[i];
    vertex_1 = varying_dim[num_points - (i + 1)];

    range_val_0 =
        ranges_vec[range_index_on_device(vertex_0, dim_index, dims_vec)];
    range_val_1 =
        ranges_vec[range_index_on_device(vertex_1, dim_index, dims_vec)];

    eval_point_0 = vertex_func_evals[i];
    eval_point_1 = vertex_func_evals[num_points - (i + 1)];

    output_evals[i] = linear_interp(interp_points[dim_index], range_val_0,
                                       range_val_1, eval_point_0, eval_point_1);

    output_vertices[i] = input_vertices[i];

    // if (particle_count == 0) {
    //   printf("dim_index: %d\n", dim_index);
    //   printf("interp_point: %e\n", interp_points.at(0));
    //   printf("vertex_0: %ld\n", vertex_0);
    //   printf("vertex_1: %ld\n", vertex_1);
    //   printf("range_val0: %e\n", range_val_0);
    //   printf("range_val1: %e\n", range_val_1);
    //   printf("eval_point_0: %e\n", eval_point_0);
    //   printf("eval_point_1: %e\n", eval_point_1);
    //   printf("output_eval_%d: %e\n", i, output_evals.at(i));
    //   printf("output_vertices: %d\n", output_vertices.at(i));
    //   printf("\n");
    // }
  }
}

} // namespace VANTAGE::Reactions::interp_utils
#endif
