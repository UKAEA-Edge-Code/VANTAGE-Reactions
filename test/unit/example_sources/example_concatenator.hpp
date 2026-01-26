inline void concatenator_example() {

  auto data_1 = FixedRateData(1.0);
  auto data_2 = FixedRateData(2.0);

  // Will return an array with the results of the contained
  // objects concatenated - [1,2] in this case
  auto concatenated_data = ConcatenatorData(data_1, data_2);

  return;
}
