void extractor_example() {

  // Extract the first 2 components (template arg) of the particle POSITION
  auto extracted_data = ExtractorData<2>(NP::Sym<REAL>("POSITION"));

  // Alternatively

  auto extracted_data_quick = extract<2>("POSITION");

  // Extract 1 component of the particle POSITION starting at offset 1 (e.g.
  // y-component)
  auto extracted_y = ExtractorData<1>(NP::Sym<REAL>("POSITION"), 1);

  // Alternatively
  auto extracted_y_quick = extract<1>("POSITION", 1);
  return;
}
