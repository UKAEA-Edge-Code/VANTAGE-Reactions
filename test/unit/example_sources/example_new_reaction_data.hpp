// Namespace containing required properties
namespace DUMMY_DATA {

const auto props =
    default_properties; // Here we use the default properties enum,
                        // but it could be any user-extended enum

// Here we only specify a single simple real prop
// This can be done for int props, as well as for species props
//
// See below for how this is used
const std::vector<int> required_simple_real_props = {props.weight};
} // namespace DUMMY_DATA

// This is the on-device type
//
// The function calc_data must be callable from a NESO-Particles
// ParticleLoop
//
// See NESO-Particles ParticleLoop documentation for details
struct DummyDataOnDevice
    : public ReactionDataBaseOnDevice<1 // This is the dimensionality of the
                                        // data In general, you would also
                                        // template against the RNG kernel type
                                        // if used (unused here)
                                      > {
  DummyDataOnDevice(REAL rate // Here we just set the fixed coefficient
                    )
      : rate(rate){};

  std::array<REAL, 1> calc_data(
      const Access::LoopIndex::Read
          &index, // This is the NESO-Particles index accessor, needed for
                  // accessing the particle data
      const Access::SymVector::Write<INT>
          &req_int_props, // These are the required integer properties (here
                          // unused)
      const Access::SymVector::Read<REAL>
          &req_real_props, // These are the required real properties - there
                           // will be only on but we will use the general
                           // indexing approach
      typename ReactionDataBaseOnDevice::RNG_KERNEL_TYPE::KernelType
          &kernel // This is the random kernel - unused here - see the
                  // FilteredMaxwellianSampler for an example where this is used
  ) const {

    auto weight = req_real_props.at(
        this->weight_ind, index, 0); // Here we access the required real prop at
                                     // the weight_ind - the particle weight

    return std::array<REAL, 1>{
        weight * this->rate}; // The rate is given as weight * rate
  }

public:
  int weight_ind; // This is the weight index that is set in the host type
  REAL rate;
};

// This is the host type
struct DummyData
    : public ReactionDataBase<1 // The data dimensionality here again
                              > {

  DummyData(REAL rate_coefficient,
            std::map<int, std::string> properties_map =
                get_default_map() // Here we allow for property remapping
            )
      : ReactionDataBase(
            Properties<REAL>(
                DUMMY_DATA::required_simple_real_props, // This is where the
                                                        // required data enums
                                                        // go in
                std::vector<Species>{},
                std::vector<int>{}), // Here no Species required properties
            properties_map),
        device_object(DummyDataOnDevice(
            rate_coefficient) // Here we call the device object constructor
        ) {

    auto props = DUMMY_DATA::props; // The used property enum

    // Here we set the weight index by calling the Properties
    // simple_prop_index() method
    //
    // See the Properties class implementation, as well as the reaction data
    // base classes
    this->device_object.weight_ind =
        this->required_real_props.simple_prop_index(
            props.weight,
            this->properties_map // Passing the map here to enable remapping
        );
  }

private:
  // The on-device data must live on the host type
  DummyDataOnDevice device_object;

public:
  // We need to provide this accessor so that the object can be retrieved for
  // use in loops
  DummyDataOnDevice get_on_device_obj() { return this->device_object; }
};
