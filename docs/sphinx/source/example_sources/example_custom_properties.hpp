struct custom_properties_enum : standard_properties_enum {
public:
  enum {
    test_custom_prop1 = default_properties.fluid_flow_speed + 1,
    test_custom_prop2
  };
};
