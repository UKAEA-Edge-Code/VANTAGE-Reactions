inline void binary_array_transform_examples() {

  // Each binary array transform data object applies a transformation to
  // the results of two other data objects
  // Unlike the unary data, binary array transforms do not by definition require
  // being part of a pipeline

  // 1D and 2D reaction data objects for the examples
  auto position_data_x = extract<1>("POSITION");
  auto position_data_xy = extract<2>("POSITION");
  auto velocity_data_xy = extract<2>("VELOCITY");

  // ------------------------
  // Doing arithmetic with ReactionData objects
  // ------------------------
  // Binary arithmetic with ReactionData objects is enabled through
  // BinaryArrayTransforms

  // The following produces a ReactionData object that returns the product of
  // the results of the two RHS objects
  auto pos_squared = position_data_xy * position_data_xy;

  // If one of the ReactionData objects is 1D, the arithmetic implementation
  // supports broadcasting onto higher dimensionality objects
  // For example, the following ReactionData object would produce a 2D reaction
  // data object that returns [pos_x^2,pos_x*pos_y]

  auto pos_times_pos_x = position_data_xy * position_data_x;

  // All 4 basic arithmetic operations [+,-,*,/] are supported between
  // ReactionData objects.

  // ------------------------
  // BinaryDotArrayTransform
  // ------------------------
  // Takes the dot product of the results of the contained data objects

  auto binary_dot_transform = BinaryDotArrayTransform<2>();

  // The following data object, when used, will calculate the dot product
  // of the position and velocity vectors (evaluated using the above extractors)
  auto dot_transform_data = BinaryArrayTransformData(
      binary_dot_transform, position_data_xy, velocity_data_xy);

  // Or, more succinctly
  auto dot_transform_data_quick =
      dot_product(position_data_xy, velocity_data_xy);

  // ------------------------
  // BinaryProjectArrayTransform and BinaryProjectNormalArrayTransform
  // ------------------------
  //
  // These transforms take in two objects, and either project the first
  // input onto the second, or onto the plane normal to it. Note that if the
  // direction vector (the second input) isn't a unit vector the projection
  // will be scaled by the square of its magnitude
  //
  // The transform expects the same dimensionality of input data as that of
  // the direction vector

  auto binary_project_transform = BinaryProjectArrayTransform<2>();
  auto binary_normal_project_transform = BinaryProjectNormalArrayTransform<2>();

  // Either of the above can be wrapped into a BinaryArrayTransformData object
  // acting on two inputs
  //
  // For example, the result of the following is equivalent to the result of
  // dot_product(position_data_xy,velocity_data_xy) * velocity_data_xy

  auto binary_project_data = BinaryArrayTransformData(
      binary_project_transform, position_data_xy, velocity_data_xy);

  return;
}
