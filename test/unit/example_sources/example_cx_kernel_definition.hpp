template <int ndim_velocity>
struct ExampleCXReactionKernelsOnDevice
    : public ReactionKernelsBaseOnDevice<1 // The number of products per parent
                                           // - here only 1 - the
                                           // charge-exchanged neutral
                                         > {
  ExampleCXReactionKernelsOnDevice() = default;
  ExampleCXReactionKernelsOnDevice(REAL target_mass, REAL projectile_mass)
      : target_mass(target_mass), projectile_mass(projectile_mass) {}

  // Each kernel hook takes the reacted weight reference, the bundled
  // SingleReactionKernelAccessors, the output species IDs, and the timestep.
  void scattering_kernel(REAL &modified_weight,
                         SingleReactionKernelAccessors &accessors,
                         const std::array<int, 1> &out_states,
                         double dt) const {

    for (int dimx = 0; dimx < ndim_velocity; dimx++) {
      accessors.descendant_products.at_real(
          accessors.index, 0, this->descendant_velocity_ind,
          dimx) = // See NESO-Particles for how these accessors work in more
                  // detail Here we access the dimx-th component of the velocity
                  // of the first product
          accessors.pre_req_data.at(
              accessors.index.get_loop_linear_index(),
              dimx); // Here we acces the dimx-th component
                     // associated with the current parent
                     // - hence the use of get_loop_linear_index
    }
  }

  void weight_kernel(REAL &modified_weight,
                     SingleReactionKernelAccessors &accessors,
                     const std::array<int, 1> &out_states, double dt) const {
    // Setting the weight of the produced particle to the entired of the reacted
    // parent weight
    accessors.descendant_products.at_real(
        accessors.index, 0, this->descendant_weight_ind, 0) = modified_weight;
  }

  void transformation_kernel(REAL &modified_weight,
                             SingleReactionKernelAccessors &accessors,
                             const std::array<int, 1> &out_states,
                             double dt) const {
    // Setting the internal state to the passed
    accessors.descendant_products.at_int(accessors.index, 0,
                                         this->descendant_internal_state_ind,
                                         0) = out_states[0];
  }

  void feedback_kernel(REAL &modified_weight,
                       SingleReactionKernelAccessors &accessors,
                       const std::array<int, 1> &out_states, double dt) const {

    std::array<REAL, ndim_velocity> k_V,
        k_Vi; // The parent particle and ion velocities
    REAL vsquared = 0.0;
    REAL visquared = 0.0;

    // Getting the velocities off the particles and the accessors.pre_req_data
    for (int vdim = 0; vdim < ndim_velocity; vdim++) {
      k_V[vdim] = accessors.req_real_props.at(this->velocity_ind,
                                              accessors.index, vdim);
      k_Vi[vdim] = accessors.pre_req_data.at(
          accessors.index.get_loop_linear_index(), vdim);
      vsquared += k_V[vdim] * k_V[vdim];
      visquared += k_Vi[vdim] * k_Vi[vdim];
    }

    // Add weight to the the projectile fluid species
    accessors.req_real_props.at(this->projectile_source_density_ind,
                                accessors.index, 0) += modified_weight;

    // Remove weight from the target fluid species
    accessors.req_real_props.at(this->target_source_density_ind,
                                accessors.index, 0) -= modified_weight;

    for (int sm_dim = 0; sm_dim < ndim_velocity; sm_dim++) {
      // Take away momentum from the target fluid species (note the units being
      // weight*amu*v_0 where v_0 is the velocity normalisation)
      accessors.req_real_props.at(this->target_source_momentum_ind,
                                  accessors.index, sm_dim) -=
          this->target_mass * modified_weight * k_Vi[sm_dim];
      // Add momentum to the projectile fluid species
      accessors.req_real_props.at(this->projectile_source_momentum_ind,
                                  accessors.index, sm_dim) +=
          this->projectile_mass * modified_weight * k_V[sm_dim];
    }

    // Remove energy from the target species
    accessors.req_real_props.at(this->target_source_energy_ind, accessors.index,
                                0) -= modified_weight * this->target_mass *
                                      visquared * 0.5; // Note the units again

    // Add energy to the projectile species
    accessors.req_real_props.at(this->projectile_source_energy_ind,
                                accessors.index, 0) +=
        modified_weight * this->projectile_mass * vsquared * 0.5;

    // Reduce parent weight by the reacted weight
    accessors.req_real_props.at(this->weight_ind, accessors.index, 0) -=
        modified_weight;
  }

public:
  // Public property indices and species masses set by the host class
  INT velocity_ind, projectile_source_density_ind, projectile_source_energy_ind,
      projectile_source_momentum_ind, target_source_density_ind,
      target_source_momentum_ind, target_source_energy_ind, weight_ind;
  INT descendant_internal_state_ind, descendant_velocity_ind,
      descendant_weight_ind;
  REAL target_mass, projectile_mass;
};

template <int ndim_velocity = 2>
struct ExampleCXReactionKernels
    : public ReactionKernelsBase<
          ExampleCXReactionKernelsOnDevice<ndim_velocity>> {
  using Base =
      ReactionKernelsBase<ExampleCXReactionKernelsOnDevice<ndim_velocity>>;

  // We use the default enums
  constexpr static auto props = default_properties;

  constexpr static std::array<int, 2> required_simple_real_props = {
      props.weight, props.velocity};

  // The CX kernels will need to contribute to the projectile and target species
  // sources
  constexpr static std::array<int, 3> required_species_real_props = {
      props.source_density, props.source_energy, props.source_momentum};

  // We are setting the internal state of the produced particle so need it
  constexpr static std::array<int, 1> required_descendant_simple_int_props = {
      props.internal_state};

  // We need the descendant particle velocity and weight access for the
  // scattering and weight kernels
  constexpr static std::array<int, 2> required_descendant_simple_real_props = {
      props.velocity, props.weight};
  ExampleCXReactionKernels(
      const Species &target_species,     // The target (ion) species
      const Species &projectile_species, // The projectile (neutral
                                         // particle/parent) species
      const PropertiesMap properties_map =
          PropertiesMap() // We allow for remaping
      )
      : Base(Properties<REAL>(
                 required_simple_real_props,
                 std::vector<Species>{target_species, projectile_species},
                 required_species_real_props),
             Properties<INT>(required_descendant_simple_int_props),
             Properties<REAL>(required_descendant_simple_real_props),
             ndim_velocity, ndim_velocity, properties_map),
        target_species(target_species), projectile_species(projectile_species) {
    // Initialize the device object and masses, then index required properties
    this->on_device_obj = ExampleCXReactionKernelsOnDevice<ndim_velocity>(
        target_species.get_mass(), projectile_species.get_mass());
    this->index_on_device_object();
  }

  void index_on_device_object() override {
    // Set input and descendant property indices from the argument pack
    this->on_device_obj->velocity_ind =
        this->argument_pack.required_real_props.find_index(
            this->get_properties_map().at(props.velocity));
    this->on_device_obj->target_source_density_ind =
        this->argument_pack.required_real_props.find_index(species_property(
            this->target_species,
            this->get_properties_map().at(props.source_density)));
    this->on_device_obj->target_source_momentum_ind =
        this->argument_pack.required_real_props.find_index(species_property(
            this->target_species,
            this->get_properties_map().at(props.source_momentum)));
    this->on_device_obj->target_source_energy_ind =
        this->argument_pack.required_real_props.find_index(species_property(
            this->target_species,
            this->get_properties_map().at(props.source_energy)));
    this->on_device_obj->projectile_source_density_ind =
        this->argument_pack.required_real_props.find_index(species_property(
            this->projectile_species,
            this->get_properties_map().at(props.source_density)));
    this->on_device_obj->projectile_source_momentum_ind =
        this->argument_pack.required_real_props.find_index(species_property(
            this->projectile_species,
            this->get_properties_map().at(props.source_momentum)));
    this->on_device_obj->projectile_source_energy_ind =
        this->argument_pack.required_real_props.find_index(species_property(
            this->projectile_species,
            this->get_properties_map().at(props.source_energy)));
    this->on_device_obj->weight_ind =
        this->argument_pack.required_real_props.find_index(
            this->get_properties_map().at(props.weight));
    this->on_device_obj->descendant_internal_state_ind =
        this->argument_pack.required_descendant_int_props.find_index(
            this->get_properties_map().at(props.internal_state));
    this->on_device_obj->descendant_velocity_ind =
        this->argument_pack.required_descendant_real_props.find_index(
            this->get_properties_map().at(props.velocity));
    this->on_device_obj->descendant_weight_ind =
        this->argument_pack.required_descendant_real_props.find_index(
            this->get_properties_map().at(props.weight));
  }

private:
  Species target_species, projectile_species;
};
