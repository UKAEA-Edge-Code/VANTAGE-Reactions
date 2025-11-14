#include "include/mock_particle_group.hpp"
#include <cstdlib>
#include <gtest/gtest.h>
#include <neso_particles.hpp>
#include <neso_particles/containers/local_array.hpp>
#include <neso_particles/containers/local_memory_interlaced.hpp>
#include <neso_particles/containers/nd_local_array.hpp>
#include <neso_particles/device_functions.hpp>
#include <neso_particles/loop/particle_loop_functions.hpp>
#include <neso_particles/particle_spec.hpp>
#include <neso_particles/typedefs.hpp>
#include <vector>

using namespace NESO::Particles;

TEST(LMI, OVERWRITE) {
  LocalMemoryInterlaced<size_t> test_overwrite_mem(2);
  LocalMemoryInterlaced<REAL> test_vector_mem(6);

  const int ndim = test_overwrite_mem.size;
  const int num_points = test_vector_mem.size;
  const int num_particles = 1e1;

  auto particle_group = create_test_particle_group(num_particles);
  particle_group->add_particle_dat(Sym<INT>("FOO"), 2);

  // particle_group->sycl_target->print_device_info();

  LocalArray<REAL> result_buf_arr(particle_group->sycl_target, num_points);

  auto result_data = result_buf_arr.get();

  for (auto &res : result_data) {
    res = 200.0;
  }

  result_buf_arr.set(result_data);

  particle_loop(
      "overwrite_test_loop", particle_group,
      [=](auto particle_index, auto test_overwrite, auto test_vector, auto result_buf,
          auto foo) {
        auto particle_count = particle_index.get_loop_linear_index();
        for (int i = 0; i < 6; i++) {
          test_vector.at(i) = REAL(i + 1);
        }

        auto test_overwrite_ptr = test_overwrite.data();
        auto test_vector_ptr = test_vector.data();
        {
          for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 2; j++) {
              test_overwrite.at(j) = i + j;
            }

            // inserting print statement here fixes it

            test_vector.at(i) =
                test_overwrite_ptr[1 * test_overwrite.get_stride()] * 1.4;
          }
        }

        foo.at(0) = test_overwrite.at(0);
        foo.at(1) = test_overwrite.at(1);

        if (particle_count == 0) {
          for (int i = 0; i < 6; i++) {
            result_buf.at(i) = test_vector.at(i);
          }
        }
      },
      Access::read(ParticleLoopIndex{}), Access::write(test_overwrite_mem),
      Access::write(test_vector_mem), Access::write(result_buf_arr),
      Access::write(Sym<INT>("FOO")))
      ->execute();

  particle_group->print(Sym<INT>("FOO"));

  result_data = result_buf_arr.get();

  for (int j = 0; j < num_points; j++) {
    printf("%f\n", result_data[j]);
  }

  particle_group->domain->mesh->free();
}

TEST(ParticleLoop, local_memory_interlaced) {
  auto A = create_test_particle_group(4095);
  A->add_particle_dat(Sym<REAL>("OUT_REAL"), 7);
  A->add_particle_dat(Sym<INT>("OUT_INT"), 3);

  auto domain = A->domain;
  auto mesh = domain->mesh;
  const int cell_count = mesh->get_cell_count();
  auto sycl_target = A->sycl_target;

  LocalMemoryInterlaced<REAL> local_mem_temp(2);
  LocalMemoryInterlaced<REAL> local_mem_real(7);
  auto local_mem_int = std::make_shared<LocalMemoryInterlaced<INT>>(3);

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();

  particle_loop(
      A,
      [=](auto INDEX, auto ID, auto OUT_REAL, auto OUT_INT, auto LM_REAL,
          auto LM_INT, auto LM_TEMP) {
        const auto index = INDEX.get_local_linear_index();
        REAL *ptr_real = LM_REAL.data();
        INT *ptr_int = LM_INT.data();
        {
          for (int cx = 0; cx < 14; cx++) {
            LM_TEMP.at(cx % 2) = (index * 7 + (cx % 7));

            if (cx % 2 == 1) {
              LM_REAL.at(cx % 7) = LM_TEMP.at(1);
            }
          }
          for (int cx = 0; cx < 3; cx++) {
            LM_INT.at(cx) = index * 3 + cx;
          }
        }
        ID.at(0) = index;
        {
          for (int cx = 0; cx < 7; cx++) {
            OUT_REAL.at(cx) = ptr_real[cx * LM_REAL.get_stride()];
          }
          for (int cx = 0; cx < 3; cx++) {
            OUT_INT.at(cx) = ptr_int[cx * LM_INT.get_stride()];
          }
        }

        // NESO_KERNEL_ASSERT(
        //     static_cast<std::size_t>(&LM_REAL.at(1) - &LM_REAL.at(0)) ==
        //         LM_REAL.get_stride(),
        //     k_ep);
      },
      Access::read(ParticleLoopIndex{}), Access::write(Sym<INT>("ID")),
      Access::write(Sym<REAL>("OUT_REAL")), Access::write(Sym<INT>("OUT_INT")),
      Access::write(local_mem_real), Access::write(local_mem_int),
      Access::write(local_mem_temp))
      ->execute();

  ASSERT_FALSE(ep.get_flag());

  for (int cellx = 0; cellx < cell_count; cellx++) {
    auto out_real = A->get_dat(Sym<REAL>("OUT_REAL"))->cell_dat.get_cell(cellx);
    auto out_int = A->get_dat(Sym<INT>("OUT_INT"))->cell_dat.get_cell(cellx);
    auto index = A->get_dat(Sym<INT>("ID"))->cell_dat.get_cell(cellx);
    const int nrow = out_real->nrow;
    // for each particle in the cell
    for (int rowx = 0; rowx < nrow; rowx++) {
      auto idx = index->at(rowx, 0);
      for (int cx = 0; cx < 7; cx++) {
        ASSERT_NEAR(out_real->at(rowx, cx), idx * 7 + cx, 1.0e-15);
      }
      for (int cx = 0; cx < 3; cx++) {
        ASSERT_NEAR(out_int->at(rowx, cx), idx * 3 + cx, 1.0e-15);
      }
    }
  }

  A->free();
  sycl_target->free();
  mesh->free();
}
