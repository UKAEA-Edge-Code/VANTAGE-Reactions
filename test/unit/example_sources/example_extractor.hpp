inline void extractor_example() {

  // Extract the first 2 components (template arg) of the particle POSITION
  auto extracted_data = ExtractorData<2>(Sym<REAL>("POSITION"));

  // Alternatively

  auto extracted_data_quick = extract<2>("POSITION");
  return;
}
