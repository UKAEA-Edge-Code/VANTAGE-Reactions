inline void
transformation_wrapper_example(ParticleGroupSharedPtr particle_group) {

  // A transformation wrapper can be constructed with a vector of marking
  // strategies or they can be added later.
  //
  // However, it always requires a transformation strategy at construction
  //
  // The wrapper below encapsulates the instruction "remove all particles with
  // weights < 1e6"
  auto wrapper = std::make_shared<TransformationWrapper>(
      std::vector<std::shared_ptr<MarkingStrategy>>{
          make_direct_marking_strategy(
              "low_weight", [](auto w) { return w[0] < 1e-6; },
              Access::read(Sym<REAL>("WEIGHT")))},
      make_transformation_strategy<SimpleRemovalTransformationStrategy>());

  // Wrappers can be copied and/or extended
  //
  // For example, maybe we want to remove both particles with ID=0 and
  // ID=1
  //
  // For this we create 2 wrappers using the above as a base

  auto wrapper_ID0 = std::make_shared<TransformationWrapper>(*wrapper);
  auto wrapper_ID1 = std::make_shared<TransformationWrapper>(*wrapper);

  // We can then add different further marking conditions to them

  wrapper_ID0->add_marking_strategy(
      VANTAGE::Reactions::make_direct_marking_strategy(
          "id0", [](auto id) { return id[0] == 0; },
          Access::read(Sym<INT>("ID"))));

  wrapper_ID1->add_marking_strategy(
      VANTAGE::Reactions::make_direct_marking_strategy(
          "id1", [](auto id) { return id[0] == 1; },
          Access::read(Sym<INT>("ID"))));

  // Wrappers, unlike strategies, can act on both groups or subgroups
  //
  // Subselection is assumed to be performed by the successive
  // application of MarkingStrategies
  wrapper_ID0->transform(particle_group);
  wrapper_ID1->transform(particle_group);

  return;
}
