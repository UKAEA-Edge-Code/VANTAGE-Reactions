#include <reactions.hpp>
#include <gtest/gtest.h>
#include "transformation_wrapper_utils.hpp"

using namespace NESO::Particles;
using namespace Reactions;

TEST(TransformationWrapper, CompositeTransformZeroer) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group_marking(N_total);

  auto composite = std::make_shared<CompositeTransform>(
      std::vector<std::shared_ptr<TransformationStrategy>>{
          make_transformation_strategy<ParticleDatZeroer<REAL>>(
              std::vector<std::string>{"V"})});
  auto zeroerID = make_transformation_strategy<ParticleDatZeroer<INT>>(
      std::vector<std::string>{"ID"});

  composite->add_transformation(zeroerID);

  auto test_wrapper = TransformationWrapper(
      std::dynamic_pointer_cast<TransformationStrategy>(composite));
  test_wrapper.transform(particle_group);

  auto num_cells = particle_group->domain->mesh->get_cell_count();

  for (int cellx = 0; cellx < num_cells; cellx++) {
    auto id = particle_group->get_cell(Sym<INT>("ID"), cellx);
    auto V = particle_group->get_cell(Sym<REAL>("V"), cellx);
    int nrow = id->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_EQ(id->at(rowx, 0), 0.0);
      EXPECT_DOUBLE_EQ(V->at(rowx, 0), 0.0);
      EXPECT_DOUBLE_EQ(V->at(rowx, 1), 0.0);
    };
  };

  particle_group->domain->mesh->free();
}