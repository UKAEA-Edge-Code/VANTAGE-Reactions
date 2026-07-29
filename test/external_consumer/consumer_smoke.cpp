// External-consumer smoke test.
//
// This TU deliberately has NO access to the VANTAGE-Reactions source tree. It
// includes only the installed public header and links the installed compiled
// library. It proves: (a) find_package(VANTAGE-Reactions) succeeds against the
// install tree, (b) the installed CMake config resolves the imported target and
// its transitive deps (MPI, NESO-Particles, SYCL), (c) the consumer TU compiles
// against the extern-template declarations in extern_templates.hpp without
// demanding implicit instantiation, and (d) the link resolves against
// libVANTAGE-Reactions.so. It is intentionally a link/build smoke test, not a
// physics test: constructing a reaction would require a SYCL device and MPI
// setup, which is out of scope for proving the runtime library is consumable.
//
// Every instantiation the library ships (see
// src/reactions_lib/instantiations/instantiations.cpp and the matching
// `extern template` declarations in src/reactions_lib/extern_templates.hpp) is
// named below. ODR-using the type (even via a pointer) requires the class
// template to be complete, which forces the consumer TU to respect the
// extern-template declaration and let the library provide the definition. If
// the library did not ship a given instantiation the link would fail with an
// unresolved symbol at the point of ODR-use; the extern-template decl keeps
// that ODR-use from emitting the symbol here.

#include <reactions/reactions.hpp>

#include <iostream>

int main() {
  using namespace VANTAGE::Reactions;

  // --- Reactions used in the examples --------------------------------------
  using R0 = LinearReactionBase<1, FixedRateData, CXReactionKernels<2>,
                                DataCalculator<FixedRateData, FixedRateData>>;
  R0 *p0 = nullptr;
  (void)p0;

  using R1 = LinearReactionBase<0, FixedRateData, IoniseReactionKernels<2>,
                                DataCalculator<FixedRateData>>;
  R1 *p1 = nullptr;
  (void)p1;

  using R2 = LinearReactionBase<
      1, FixedRateData, RecombReactionKernels<2, 2>,
      DataCalculator<FixedRateData, FixedRateData, FixedRateData>>;
  R2 *p2 = nullptr;
  (void)p2;

  using R3 =
      LinearReactionBase<1, FixedRateData, LinearScatteringKernels<2, true>,
                         ScatteringDataCalculator>;
  R3 *p3 = nullptr;
  (void)p3;

  using R4 = LinearReactionBase<
      1, FixedRateData, CXReactionKernels<3>,
      DataCalculator<FixedRateData, FixedRateData, FixedRateData>>;
  R4 *p4 = nullptr;
  (void)p4;

  using R5 =
      LinearReactionBase<1, FixedRateData, LinearScatteringKernels<3, true>,
                         ScatteringDataCalculator3D>;
  R5 *p5 = nullptr;
  (void)p5;

  using R6 = ElectronImpactIonisation<FixedRateData, FixedRateData, 2>;
  R6 *p6 = nullptr;
  (void)p6;

  using R7 = ElectronImpactIonisation<AMJUEL1DData<9>, FixedRateData, 2>;
  R7 *p7 = nullptr;
  (void)p7;

  using R8 =
      Recombination<FixedRateData,
                    DataCalculator<FixedRateData, FixedRateData, FixedRateData>,
                    2>;
  R8 *p8 = nullptr;
  (void)p8;

  using R9 = LinearReactionBase<0, FixedRateData, GeneralAbsorptionKernels<2>>;
  R9 *p9 = nullptr;
  (void)p9;

  using R10 =
      LinearReactionBase<0, FixedRateData, SpecularReflectionKernels<2>>;
  R10 *p10 = nullptr;
  (void)p10;

  using R11 =
      LinearReactionBase<1, FixedRateData, LinearScatteringKernels<2, false>,
                         ScatteringDataCalculator>;
  R11 *p11 = nullptr;
  (void)p11;

  using R12 = LinearReactionBase<1, FixedRateData, CXReactionKernels<2>,
                                 DataCalculator<FilteredMaxwellianSampler<2>>>;
  R12 *p12 = nullptr;
  (void)p12;

  using R13 = LinearReactionBase<1, FixedRateData, CXReactionKernels<3>,
                                 DataCalculator<FilteredMaxwellianSampler<3>>>;
  R13 *p13 = nullptr;
  (void)p13;

  using R14 =
      LinearReactionBase<1, FixedRateData, LinearScatteringKernels<3, true>,
                         ScatteringDataCalculatorSpherical>;
  R14 *p14 = nullptr;
  (void)p14;

  using R15 =
      LinearReactionBase<1, FixedRateData, LinearScatteringKernels<3, true>,
                         ScatteringDataCalculatorCartesian>;
  R15 *p15 = nullptr;
  (void)p15;

  using R16 = Recombination<FixedRateData,
                            DataCalculator<FixedRateData, FixedRateData,
                                           FixedRateData, FixedRateData>,
                            3>;
  R16 *p16 = nullptr;
  (void)p16;

  using R17 = ElectronImpactIonisation<FixedRateData, FixedRateData, 3>;
  R17 *p17 = nullptr;
  (void)p17;

  // --- Kernel classes ------------------------------------------------------
  using K0 = CXReactionKernels<2>;
  K0 *k0 = nullptr;
  (void)k0;

  using K1 = IoniseReactionKernels<2>;
  K1 *k1 = nullptr;
  (void)k1;

  using K2 = RecombReactionKernels<2, 2>;
  K2 *k2 = nullptr;
  (void)k2;

  using K3 = LinearScatteringKernels<2, true>;
  K3 *k3 = nullptr;
  (void)k3;

  using K4 = CXReactionKernels<3>;
  K4 *k4 = nullptr;
  (void)k4;

  using K5 = LinearScatteringKernels<3, true>;
  K5 *k5 = nullptr;
  (void)k5;

  using K6 = GeneralAbsorptionKernels<2>;
  K6 *k6 = nullptr;
  (void)k6;

  using K7 = SpecularReflectionKernels<2>;
  K7 *k7 = nullptr;
  (void)k7;

  using K8 = LinearScatteringKernels<2, false>;
  K8 *k8 = nullptr;
  (void)k8;

  using K9 = GeneralAbsorptionKernels<3>;
  K9 *k9 = nullptr;
  (void)k9;

  using K10 = SpecularReflectionKernels<3>;
  K10 *k10 = nullptr;
  (void)k10;

  using K11 = LinearScatteringKernels<3, false>;
  K11 *k11 = nullptr;
  (void)k11;

  using K12 = IoniseReactionKernels<3>;
  K12 *k12 = nullptr;
  (void)k12;

  using K13 = RecombReactionKernels<3, 3>;
  K13 *k13 = nullptr;
  (void)k13;

  // --- DataCalculator specialisations --------------------------------------
  using D0 = DataCalculator<FixedRateData>;
  D0 *d0 = nullptr;
  (void)d0;

  using D1 = DataCalculator<FixedRateData, FixedRateData>;
  D1 *d1 = nullptr;
  (void)d1;

  using D2 = DataCalculator<FixedRateData, FixedRateData, FixedRateData>;
  D2 *d2 = nullptr;
  (void)d2;

  using D3 = DataCalculator<VelocityReflectionPipeline>;
  D3 *d3 = nullptr;
  (void)d3;

  using D4 = DataCalculator<FilteredMaxwellianSampler<2>>;
  D4 *d4 = nullptr;
  (void)d4;

  using D5 = DataCalculator<FilteredMaxwellianSampler<3>>;
  D5 *d5 = nullptr;
  (void)d5;

  using D6 = DataCalculator<FixedRateData, FixedRateData, FixedRateData,
                            FixedRateData>;
  D6 *d6 = nullptr;
  (void)d6;

  using D7 = DataCalculator<SphericalReflectionPipeline>;
  D7 *d7 = nullptr;
  (void)d7;

  using D8 = DataCalculator<CartesianReflectionPipeline>;
  D8 *d8 = nullptr;
  (void)d8;

  using D9 = DataCalculator<VelocityReflectionPipeline3D>;
  D9 *d9 = nullptr;
  (void)d9;

  // --- Common transformation strategies ------------------------------------
  using T0 = CellwiseAccumulator<REAL>;
  T0 *t0 = nullptr;
  (void)t0;

  using T1 = WeightedCellwiseAccumulator<REAL>;
  T1 *t1 = nullptr;
  (void)t1;

  using T2 = ParticleDatZeroer<REAL>;
  T2 *t2 = nullptr;
  (void)t2;

  using T3 = MergeTransformationStrategy<2>;
  T3 *t3 = nullptr;
  (void)t3;

  using T4 = CellwiseAccumulator<INT>;
  T4 *t4 = nullptr;
  (void)t4;

  using T5 = WeightedCellwiseAccumulator<INT>;
  T5 *t5 = nullptr;
  (void)t5;

  using T6 = ParticleDatZeroer<INT>;
  T6 *t6 = nullptr;
  (void)t6;

  using T7 = CellwiseDistributor<REAL>;
  T7 *t7 = nullptr;
  (void)t7;

  using T8 = CellwiseDistributor<INT>;
  T8 *t8 = nullptr;
  (void)t8;

  using T9 = MergeTransformationStrategy<3>;
  T9 *t9 = nullptr;
  (void)t9;

  // --- Downsampling strategies ---------------------------------------------
  using S0 = VranicMergingKernels<2>;
  S0 *s0 = nullptr;
  (void)s0;

  using S1 = DownsamplingStrategy<VranicMergingKernels<2>>;
  S1 *s1 = nullptr;
  (void)s1;

  using S2 = DownsamplingStrategy<SimpleThinningKernels>;
  S2 *s2 = nullptr;
  (void)s2;

  using S3 = VranicMergingKernels<3>;
  S3 *s3 = nullptr;
  (void)s3;

  using S4 = DownsamplingStrategy<VranicMergingKernels<3>>;
  S4 *s4 = nullptr;
  (void)s4;

  // --- Other heavy reaction data types -------------------------------------
  using H0 = FilteredMaxwellianSampler<2, ConstantRateCrossSection>;
  H0 *h0 = nullptr;
  (void)h0;

  using H1 = CellwiseReactionDataAccumulator<KinEnergyData>;
  H1 *h1 = nullptr;
  (void)h1;

  using H2 = AMJUEL1DData<3>;
  H2 *h2 = nullptr;
  (void)h2;

  using H3 = AMJUEL1DData<9>;
  H3 *h3 = nullptr;
  (void)h3;

  using H4 = AMJUEL2DData<2, 2>;
  H4 *h4 = nullptr;
  (void)h4;

  using H5 = AMJUEL2DDataH3<2, 2>;
  H5 *h5 = nullptr;
  (void)h5;

  using H6 = FixedArrayData<3>;
  H6 *h6 = nullptr;
  (void)h6;

  using H7 = ArrayLookupData<1>;
  H7 *h7 = nullptr;
  (void)h7;

  using H8 = ArrayLookupData<1, true>;
  H8 *h8 = nullptr;
  (void)h8;

  using H9 = AMJUELFitCrossSection<2, 0, 0>;
  H9 *h9 = nullptr;
  (void)h9;

  using H10 = AMJUELFitCrossSection<2, 2, 0>;
  H10 *h10 = nullptr;
  (void)h10;

  using H11 = AMJUELFitCrossSection<2, 2, 2>;
  H11 *h11 = nullptr;
  (void)h11;

  using H12 = AMJUELFitCrossSection<3, 3, 3>;
  H12 *h12 = nullptr;
  (void)h12;

  using H13 = FilteredMaxwellianSampler<3>;
  H13 *h13 = nullptr;
  (void)h13;

  using H14 = ExtractorData<1>;
  H14 *h14 = nullptr;
  (void)h14;

  using H15 = ExtractorData<2>;
  H15 *h15 = nullptr;
  (void)h15;

  using H16 = ExtractorData<3>;
  H16 *h16 = nullptr;
  (void)h16;

  using H17 = SpecularReflectionData<2>;
  H17 *h17 = nullptr;
  (void)h17;

  using H18 = SpecularReflectionData<3>;
  H18 *h18 = nullptr;
  (void)h18;

  // --- Interpolation / grid family -----------------------------------------
  using G0 = CartesianGridData<1>;
  G0 *g0 = nullptr;
  (void)g0;

  using G1 = CartesianGridData<2>;
  G1 *g1 = nullptr;
  (void)g1;

  using G2 = CartesianGridData<3>;
  G2 *g2 = nullptr;
  (void)g2;

  using G3 = CartesianGridData<4>;
  G3 *g3 = nullptr;
  (void)g3;

  using G4 = CartesianGridData<5>;
  G4 *g4 = nullptr;
  (void)g4;

  using G5 = TrimEvalData<5>;
  G5 *g5 = nullptr;
  (void)g5;

  using G6 = TrimEvalData<7>;
  G6 *g6 = nullptr;
  (void)g6;

  std::cout << "consumer_smoke: linked against libVANTAGE-Reactions.so OK\n";
  return 0;
}
