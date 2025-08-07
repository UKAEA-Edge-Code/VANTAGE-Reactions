#include <reactions.hpp>
#include <gtest/gtest.h>
#include "transformation_wrapper_utils.hpp"

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

TEST(TransformationWrapper, SimpleRemovalTransformationStrategy_less_than) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group_marking(N_total);

  auto test_wrapper = TransformationWrapper(
      std::vector<std::shared_ptr<MarkingStrategy>>{
          make_marking_strategy<ComparisonMarkerSingle<REAL, LessThanComp>>(
              Sym<REAL>("WEIGHT"), 0.5)},
      make_transformation_strategy<SimpleRemovalTransformationStrategy>());

  test_wrapper.transform(particle_group);

  auto num_cells = particle_group->domain->mesh->get_cell_count();

  for (int cellx = 0; cellx < num_cells; cellx++) {
    auto W = particle_group->get_cell(Sym<REAL>("WEIGHT"), cellx);
    int nrow = W->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(W->at(rowx, 0), 1.0);
    };
  };

  particle_group->domain->mesh->free();
}