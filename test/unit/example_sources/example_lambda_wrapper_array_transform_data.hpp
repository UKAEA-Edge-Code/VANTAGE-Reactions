inline void lambda_wrapper_array_transform_examples() {

  // 1D and 2D reaction data objects for the examples
  auto position_data_x = extract<1>("POSITION");
  auto position_data_xy = extract<2>("POSITION");
  auto velocity_data_xy = extract<2>("VELOCITY");

  // Supported lambdas are binary and unary functions of either
  // conforming REAL arrays or of REAL values, allowing for full array or
  // elementwise application

  // ------------------
  // Binary full array
  // ------------------

  auto binary_lambda_full = [](const std::array<REAL, 2> &a,
                               const std::array<REAL, 2> &b) {
    return std::array<REAL, 2>{a[0] * b[1], b[1]};
  };

  // The lambda wrapper is templated on the type of the lambda, as well as the
  // expected array dimension
  auto lambda_wrapper_binary_full =
      utils::LambdaWrapper<decltype(binary_lambda_full), 2>{binary_lambda_full};

  // batData is a helper function for turning lambda wrappers into
  // full array binary transform data object
  auto full_binary_transform_data =
      batData(lambda_wrapper_binary_full, position_data_xy, velocity_data_xy);

  // ------------------
  // Binary elementwise
  // ------------------

  auto binary_lambda_elementwise = [](const REAL &a, const REAL &b) {
    return 2 * a + b;
  };

  auto lambda_wrapper_elementwise =
      utils::LambdaWrapper<decltype(binary_lambda_elementwise), 1>(
          binary_lambda_elementwise);
  // betData is a helper function for turning lambda wrappers into elementwise
  // binary transform data objects
  auto elementwise_binary_transform_data =
      betData(lambda_wrapper_elementwise, position_data_xy, position_data_xy);

  // ------------------
  // Unary full array
  // ------------------

  auto unary_lambda_full = [=](const std::array<REAL, 2> &a) {
    return std::array<REAL, 2>{a[0] * a[0], a[1] * a[1]};
  };

  auto unary_lambda_wrapper_full =
      utils::LambdaWrapper<decltype(unary_lambda_full), 2>(unary_lambda_full);

  // uatData is a helper function templated on the dimensionality of the data
  // and the type of the lambda generating unary full array transform data from
  // a unary lambda
  auto full_unary_transform_data =
      uatData<2, decltype(unary_lambda_wrapper_full)>(
          unary_lambda_wrapper_full);

  // As with regular unary array transforms, these must be in a pipeline to be
  // used as standard ReactionData
  auto unary_lambda_pipeline =
      pipe(position_data_xy, full_unary_transform_data);

  // ------------------
  // Unary elementwise
  // ------------------

  auto unary_lambda_elementwise = [](const REAL &a) { return 2 * a; };

  auto unary_lambda_wrapper_elementwise =
      utils::LambdaWrapper(unary_lambda_elementwise);

  // uetData is a helper function generating elementwise unary transform data
  // from a lambda
  auto elementwise_unary_transform_data =
      uetData<2, decltype(unary_lambda_wrapper_elementwise)>(
          unary_lambda_wrapper_elementwise);

  // Same as above - we need a pipeline in order to make use of unary array
  // transform data
  auto unary_lambda_pipeline_elementwise =
      pipe(position_data_xy, elementwise_unary_transform_data);

  return;
}
