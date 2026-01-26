inline void fixed_array_data_example() {

  auto data_array = std::array<REAL, 3>{1.0, 2.0, 3.0};
  // The following will just return the above array
  // NOTE: templated on array size so can return any length array
  auto fixed_array = FixedArrayData(data_array);

  return;
}
