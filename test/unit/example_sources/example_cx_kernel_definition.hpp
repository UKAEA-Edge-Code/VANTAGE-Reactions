template <int ndim_velocity>
struct ExampleCXReactionKernelsOnDevice
    : public ReactionKernelsBaseOnDevice<
          1 // The number of products per parent - here only 1 - the charge-exchanged neutral 
      > {
  ExampleCXReactionKernelsOnDevice() = default;

  // All kernels have the same signature
  void scattering_kernel(
      REAL &modified_weight, // This is the weight involved in the reaction (this is calculated by the base reaction class)
      Access::LoopIndex::Read &index, // Indexing object used to access
      Access::DescendantProducts::Write &descendant_products, // The accessor for descendant product properties
      Access::SymVector::Write<INT> &req_int_props, // Accessor for parent particle required integer properties
      Access::SymVector::Write<REAL> &req_real_props, // Accessor for parent particle required real properties
      const std::array<int, 1> &out_states, // The outgoing state indices 
      Access::NDLocalArray::Read<REAL, 2> &pre_req_data, // Pre-calculated data from the DataCalculators
      double dt // The timestep taken 
      ) const {

    for (int dimx = 0; dimx < ndim_velocity; dimx++) {
      descendant_products.at_real(index, 0, descendant_velocity_ind, dimx) = // See NESO-Particles for how these accessors work in more detail 
                                                                             // Here we access the dimx-th component of the velocity of the first product
          pre_req_data.at(index.get_loop_linear_index(), dimx); // Here we acces the dimx-th component associated with the current parent 
                                                                // - hence the use of get_loop_linear_index
    }
  }

  void
  weight_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                Access::DescendantProducts::Write &descendant_products,
                Access::SymVector::Write<INT> &req_int_props,
                Access::SymVector::Write<REAL> &req_real_props,
                const std::array<int, 1>
                    &out_states,
                Access::NDLocalArray::Read<REAL, 2> &pre_req_data,
                double dt) const {
    // Setting the weight of the produced particle to the entired of the reacted parent weight
    descendant_products.at_real(index, 0, descendant_weight_ind, 0) =
        modified_weight;
  }

  void transformation_kernel(
      REAL &modified_weight, Access::LoopIndex::Read &index,
      Access::DescendantProducts::Write &descendant_products,
      Access::SymVector::Write<INT> &req_int_props,
      Access::SymVector::Write<REAL> &req_real_props,
      const std::array<int, 1>
          &out_states,
      Access::NDLocalArray::Read<REAL, 2> &pre_req_data, double dt) const {
    // Setting the internal state to the passed 
    descendant_products.at_int(index, 0, descendant_internal_state_ind, 0) =
        out_states[0];
  }

  void
  feedback_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                  Access::DescendantProducts::Write &descendant_products,
                  Access::SymVector::Write<INT> &req_int_props,
                  Access::SymVector::Write<REAL> &req_real_props,
                  const std::array<int, 1>
                      &out_states,
                  Access::NDLocalArray::Read<REAL, 2> &pre_req_data,
                  double dt) const {

    std::array<REAL, ndim_velocity> k_V, k_Vi; // The parent particle and ion velocities
    REAL vsquared = 0.0;
    REAL visquared = 0.0;

    // Getting the velocities off the particles and the pre_req_data
    for (int vdim = 0; vdim < ndim_velocity; vdim++) {
      k_V[vdim] = req_real_props.at(velocity_ind, index, vdim);
      k_Vi[vdim] = pre_req_data.at(index.get_loop_linear_index(), vdim); 
      vsquared += k_V[vdim] * k_V[vdim];
      visquared += k_Vi[vdim] * k_Vi[vdim];
    }

    // Add weight to the the projectile fluid species
    req_real_props.at(this->projectile_source_density_ind, index, 0) +=
        modified_weight;

    // Remove weight from the target fluid species
    req_real_props.at(this->target_source_density_ind, index, 0) -=
        modified_weight;

    for (int sm_dim = 0; sm_dim < ndim_velocity; sm_dim++) {
      // Take away momentum from the target fluid species (note the units being weight*amu*v_0 where v_0 is the velocity normalisation) 
      req_real_props.at(this->target_source_momentum_ind, index, sm_dim) -=
          this->target_mass * modified_weight * k_Vi[sm_dim];
      // Add momentum to the projectile fluid species
      req_real_props.at(this->projectile_source_momentum_ind, index, sm_dim) +=
          this->projectile_mass * modified_weight * k_V[sm_dim];
    }

    // Remove energy from the target species
    req_real_props.at(this->target_source_energy_ind, index, 0) -=
        modified_weight * this->target_mass * visquared * 0.5; // Note the units again

    // Add energy to the projectile species
    req_real_props.at(this->projectile_source_energy_ind, index, 0) +=
        modified_weight * this->projectile_mass * vsquared * 0.5;

    // Reduce parent weight by the reacted weight
    req_real_props.at(this->weight_ind, index, 0) -= modified_weight;
  }

public:
  // Public data and indices set in the host constructor (see below)
  INT velocity_ind, projectile_source_density_ind, projectile_source_energy_ind,
      projectile_source_momentum_ind, target_source_density_ind,
      target_source_momentum_ind, target_source_energy_ind, weight_ind;
  INT descendant_internal_state_ind, descendant_velocity_ind,
      descendant_weight_ind;
  REAL target_mass, projectile_mass;
};

template <int ndim_velocity = 2>
struct ExampleCXReactionKernels : public ReactionKernelsBase {

    // We use the default enums
    constexpr static auto props = default_properties;

    constexpr static std::array<int,2> required_simple_real_props = {props.weight,
                                                         props.velocity};

    // The CX kernels will need to contribute to the projectile and target species sources
    constexpr static std::array<int,3> required_species_real_props = {
        props.source_density, props.source_energy,
        props.source_momentum}; 

    // We are setting the internal state of the produced particle so need it
    constexpr static std::array<int,1> required_descendant_simple_int_props = {
        props.internal_state};

    // We need the descendant particle velocity and weight access for the scattering and weight kernels
    constexpr static std::array<int,2> required_descendant_simple_real_props = {props.velocity,
                                                                    props.weight};
  ExampleCXReactionKernels(const Species &target_species, // The target (ion) species
                    const Species &projectile_species, // The projectile (neutral particle/parent) species
                    std::map<int, std::string> properties_map = get_default_map() // We allow for remaping
                    )
      : ReactionKernelsBase(
            Properties<REAL>( // The Properties container object for the required properties on the parent
                required_simple_real_props,
                std::vector<Species>{target_species, projectile_species},
                required_species_real_props),
            ndim_velocity, properties_map) {

    // Here we set all of the required indices for the various properties on device
    this->cx_reaction_kernels_on_device.velocity_ind =
        this->required_real_props.simple_prop_index(props.velocity,
                                                    this->properties_map);

    this->cx_reaction_kernels_on_device.target_source_density_ind =
        this->required_real_props.species_prop_index(target_species.get_name(),
                                                     props.source_density,
                                                     this->properties_map);

    this->cx_reaction_kernels_on_device.target_source_momentum_ind =
        this->required_real_props.species_prop_index(target_species.get_name(),
                                                     props.source_momentum,
                                                     this->properties_map);

    this->cx_reaction_kernels_on_device.target_source_energy_ind =
        this->required_real_props.species_prop_index(target_species.get_name(),
                                                     props.source_energy,
                                                     this->properties_map);

    this->cx_reaction_kernels_on_device.projectile_source_density_ind =
        this->required_real_props.species_prop_index(
            projectile_species.get_name(), props.source_density,
            this->properties_map);

    this->cx_reaction_kernels_on_device.projectile_source_momentum_ind =
        this->required_real_props.species_prop_index(
            projectile_species.get_name(), props.source_momentum,
            this->properties_map);

    this->cx_reaction_kernels_on_device.projectile_source_energy_ind =
        this->required_real_props.species_prop_index(
            projectile_species.get_name(), props.source_energy,
            this->properties_map);

    this->cx_reaction_kernels_on_device.weight_ind =
        this->required_real_props.simple_prop_index(props.weight,
                                                    this->properties_map);

    this->cx_reaction_kernels_on_device.target_mass = target_species.get_mass();
    this->cx_reaction_kernels_on_device.projectile_mass =
        projectile_species.get_mass();

    // These set the descendant particle properties
    this->set_required_descendant_int_props(
        Properties<INT>(required_descendant_simple_int_props));

    this->set_required_descendant_real_props(Properties<REAL>(
       required_descendant_simple_real_props));

    // Here we set the descendant particle indices based on the above properties
    this->cx_reaction_kernels_on_device.descendant_internal_state_ind =
        this->required_descendant_int_props.simple_prop_index(
            props.internal_state, this->properties_map);
    this->cx_reaction_kernels_on_device.descendant_velocity_ind =
        this->required_descendant_real_props.simple_prop_index(
            props.velocity, this->properties_map);
    this->cx_reaction_kernels_on_device.descendant_weight_ind =
        this->required_descendant_real_props.simple_prop_index(
            props.weight, this->properties_map);

    // Finally, we set up the descendant matrix spec
    this->set_descendant_matrix_spec<ndim_velocity,
                                     1>();
  };

private:
  ExampleCXReactionKernelsOnDevice<ndim_velocity>
      cx_reaction_kernels_on_device;

public:
  /**
   * @brief Getter for the SYCL device-specific struct.
   */

  ExampleCXReactionKernelsOnDevice<ndim_velocity>
  get_on_device_obj() {
    return this->cx_reaction_kernels_on_device;
  }
};
