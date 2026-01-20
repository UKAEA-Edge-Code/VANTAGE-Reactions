inline void unary_array_transform_examples() {

  // Each unary array transform data object applies a transformation to
  // an input array

  // This is just some dummy data for the example
  auto position_data = extract<2>("POSITION");

  // ------------------------
  // PolynomialArrayTransform
  // ------------------------
  // The following transformation takes an array of polynomial
  // coefficients and applies them to the input.
  //
  // The two template arguments are the expected input dimension and the
  // polynomial order (1 less than the dimension of the coefficients)
  //
  // In this case, the elementwise polynomial will be 2x + 1
  auto linear_poly =
      PolynomialArrayTransform<2, 1>(std::array<REAL, 2>{1.0, 2.0});

  // The transform can be wrapped into a reaction data object
  auto linear_poly_data = UnaryArrayTransformData(linear_poly);

  // And the object can have the input data piped into it
  auto pipe_poly = pipe(position_data, linear_poly_data);

  // ------------------------
  // ScalerArrayTransform
  // ------------------------
  // The following transform just scales the input elementwise by a number
  // The template argument is the expected input dimension

  auto scaler = ScalerArrayTransform<2>(2.0);

  auto scaler_data = UnaryArrayTransformData(scaler);

  // or more succinctly
  auto scaler_data_quick = scale_by<2>(2.0);

  // And either of the above can then be used in a pipeline.

  // ------------------------
  // UnaryProjectArrayTransform and UnaryProjectNormalArrayTransform
  // ------------------------
  //
  // These transforms take in a constant direction, and either project the input
  // onto that direction, or onto the plane normal to it. Note that if the
  // direction vector isn't a unit vector the projection will be scaled by
  // the square of its magnitude
  //
  // The transform expects the same dimensionality of input data as that of the
  // direction vector

  auto dir = std::array<REAL, 2>{1.0, 0.0};

  // Projects onto dir
  auto project = UnaryProjectArrayTransform(dir);

  // Project onto the plane normal to dir (in this case [0,1])
  auto project_normal = UnaryProjectNormalArrayTransform(dir);

  // The above can then be wrapped in UnaryArrayTransformData and used in a
  // pipeline

  return;
}
