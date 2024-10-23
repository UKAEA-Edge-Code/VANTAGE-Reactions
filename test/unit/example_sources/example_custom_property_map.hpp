inline void custom_property_map_example() {

  auto custom_props = custom_properties_enum();

  // The property map is wrapped in a class that offers basic consistency checks
  // We can initialise a custom property map with the default constructor, which
  // uses the default property mapping

  auto custom_property_map = properties_map();

  // We can then extend the map to work with our custom enum
  custom_property_map[custom_props.test_custom_prop1] = "TEST_PROP1";
  custom_property_map[custom_props.test_custom_prop2] = "TEST_PROP2";

  // We can also remap the existing properties
  custom_property_map[default_properties.weight] = "w";

  return;
}
