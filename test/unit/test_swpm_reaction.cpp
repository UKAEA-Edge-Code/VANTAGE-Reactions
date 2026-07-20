#include "include/mock_particle_group.hpp"
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

TEST(SWPMReaction, simple_hard_spheres) {
  // This is a hard-coded number of particles so we get a round number for pair
  // construction (800/16 = 40 particles per cell, which is 20 pairs)
  const int N_total = 800;

  auto A = create_test_particle_group(N_total);

  int cell_count = A->domain->mesh->get_cell_count();

  auto cellwise_pair_listA =
      std::make_shared<CellwisePairListSimple>(A->sycl_target, cell_count);

  std::vector<int> c;
  std::vector<int> i;
  std::vector<int> j;

  int npart_cell = std::round(A->get_npart_local() /
                              (double)A->domain->mesh->get_cell_count());

  c.reserve(cell_count * npart_cell / 2);
  i.reserve(cell_count * npart_cell / 2);
  j.reserve(cell_count * npart_cell / 2);

  std::mt19937 rng(9124234 + A->sycl_target->comm_pair.rank_parent);

  for (int cellx = 0; cellx < cell_count; cellx++) {
    npart_cell = A->get_npart_cell(cellx);
    std::vector<int> pairs(npart_cell);
    std::iota(pairs.begin(), pairs.end(), 0);
    std::shuffle(pairs.begin(), pairs.end(), rng);
    for (int px = 0; px < (npart_cell / 2); px++) {
      c.push_back(cellx);
      i.push_back(pairs.at(2 * px));
      j.push_back(pairs.at(2 * px + 1));
    }
  }

  cellwise_pair_listA->push_back(c, i, j);

  auto cs_data = CSPairData<2>(ConstantRateCrossSection(0.1));

  auto rng_lambda = [&]() -> REAL { return 0.6; };

  auto rng_kernel = host_atomic_block_kernel_rng<REAL>(rng_lambda, 2);

  // Mocking the species to test the correct reduced mass effects
  auto species_1 = Species("ION", 1.2, 0.0, 0);
  auto species_2 = Species("ION2", 2.0, 0.0, 1);
  auto hs_scattering_data =
      HSScatteringData<2>(species_1, species_2, rng_kernel);
  auto pair_data_calculator = PairDataCalculator(hs_scattering_data);

  auto scattering_kernels = PairScatteringKernels<2>();

  auto swpm_reaction =
      SWPMReaction<2, decltype(cs_data), decltype(scattering_kernels),
                   decltype(pair_data_calculator)>(
          A->sycl_target, std::array<int, 2>{0, 0}, std::array<int, 2>{1, 2},
          cs_data, scattering_kernels, pair_data_calculator);

  auto pair_list = CellwisePairListAbsolute<ParticleGroup, CellwisePairList>(
      A, A, cellwise_pair_listA);

  for (int i = 0; i < cell_count; i++) {
    swpm_reaction.calculate_rates(pair_list, i, i + 1);
  }
  particle_loop(
      "copy_weight_change_test", A,
      [=](auto tot_rate, auto weight_change) {
        weight_change[0] = tot_rate[0];
      },
      Access::read(Sym<REAL>("TOT_REACTION_RATE")),
      Access::write(Sym<REAL>("WEIGHT_CHANGE")))
      ->execute();

  auto descendant_particles = std::make_shared<ParticleGroup>(
      A->domain, A->get_particle_spec(), A->sycl_target);

  for (int i = 0; i < cell_count; i++) {
    swpm_reaction.apply(pair_list, i, i + 1, 1.0, descendant_particles);
  }
  A->sycl_target->free();
  A->domain->mesh->free();
}
