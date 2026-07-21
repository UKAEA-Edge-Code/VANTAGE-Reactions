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

  // --- Other heavy reaction data types -------------------------------------
  using H0 = FilteredMaxwellianSampler<2, ConstantRateCrossSection>;
  H0 *h0 = nullptr;
  (void)h0;

  using H1 = CellwiseReactionDataAccumulator<KinEnergyData>;
  H1 *h1 = nullptr;
  (void)h1;

  std::cout << "consumer_smoke: linked against libVANTAGE-Reactions.so OK\n";
  return 0;
}
