inline void pipeline_example() {

  // Here we will pipe particle velocities into a specular reflection data
  // object See the documentation of the specular reflection for more details

  // In case we wish to remap the default boundary normal for the specular
  // reflection
  auto used_map = get_default_map();

  auto velocity_data = extract<2>("VELOCITY");

  auto specular_reflection = SpecularReflectionData<2>(used_map);

  // The pipeline object will first evaluate the velocity extractor
  // and will then pipe this into the specular reflection,
  // resulting in a specularly reflected velocity (assuming that the surface
  // normal is correctly set)
  //
  // In general, this allows for more flexibility, as we might transform the
  // velocity data somehow before passing it to the reflection data
  auto pipeline = PipelineData(velocity_data, specular_reflection);

  // Alternative syntax
  auto pipeline_quick = pipe(velocity_data, specular_reflection);

  return;
}
