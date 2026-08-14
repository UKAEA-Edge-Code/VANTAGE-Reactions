// External-consumer smoke test.
//
// This TU deliberately has NO access to the VANTAGE-Reactions source tree. It
// includes only the installed public header and links the installed compiled
// library. It proves: (a) find_package(VANTAGE-Reactions) succeeds against the
// install tree, (b) the installed CMake config resolves the imported target and
// its transitive deps (MPI, NESO-Particles, SYCL), (c) the consumer TU compiles
// against the extern-template declarations without demanding implicit
// instantiation, and (d) the link resolves against libVANTAGE-Reactions.so.
//
// Every instantiation the library ships (see
// src/ and the matching `extern template` declarations in
// the corresponding header files) is named below. ODR-using the
// type (even via a pointer) requires the class template to be complete, which
// forces the consumer TU to respect the extern-template declaration and let the
// library provide the definition. If the library did not ship a given
// instantiation the link would fail with an unresolved symbol at the point of
// ODR-use; the extern-template decl keeps that ODR-use from emitting the symbol
// here.

#include <reactions/reactions.hpp>

#include <iostream>

int main() {
  using namespace VANTAGE::Reactions;

  // --- Kernel classes ------------------------------------------------------
  using K0 = CXReactionKernels<2>;
  K0 *k0 = nullptr;
  (void)k0;

  using K1 = CXReactionKernels<3>;
  K1 *k1 = nullptr;
  (void)k1;

  using K2 = IoniseReactionKernels<2>;
  K2 *k2 = nullptr;
  (void)k2;

  using K3 = IoniseReactionKernels<3>;
  K3 *k3 = nullptr;
  (void)k3;

  using K4 = RecombReactionKernels<2>;
  K4 *k4 = nullptr;
  (void)k4;

  using K5 = RecombReactionKernels<3>;
  K5 *k5 = nullptr;
  (void)k5;

  using K6 = LinearScatteringKernels<2, true>;
  K6 *k6 = nullptr;
  (void)k6;

  using K7 = LinearScatteringKernels<2, false>;
  K7 *k7 = nullptr;
  (void)k7;

  using K8 = LinearScatteringKernels<3, true>;
  K8 *k8 = nullptr;
  (void)k8;

  using K9 = LinearScatteringKernels<3, false>;
  K9 *k9 = nullptr;
  (void)k9;

  using K10 = GeneralAbsorptionKernels<2>;
  K10 *k10 = nullptr;
  (void)k10;

  using K11 = GeneralAbsorptionKernels<3>;
  K11 *k11 = nullptr;
  (void)k11;

  using K12 = SpecularReflectionKernels<2>;
  K12 *k12 = nullptr;
  (void)k12;

  using K13 = SpecularReflectionKernels<3>;
  K13 *k13 = nullptr;
  (void)k13;

  // --- Common transformation strategies ------------------------------------
  using T0 = CellwiseAccumulator<REAL>;
  T0 *t0 = nullptr;
  (void)t0;

  using T1 = CellwiseAccumulator<INT>;
  T1 *t1 = nullptr;
  (void)t1;

  using T2 = WeightedCellwiseAccumulator<REAL>;
  T2 *t2 = nullptr;
  (void)t2;

  using T3 = WeightedCellwiseAccumulator<INT>;
  T3 *t3 = nullptr;
  (void)t3;

  using T4 = ParticleDatZeroer<REAL>;
  T4 *t4 = nullptr;
  (void)t4;

  using T5 = ParticleDatZeroer<INT>;
  T5 *t5 = nullptr;
  (void)t5;

  using T6 = MergeTransformationStrategy<2>;
  T6 *t6 = nullptr;
  (void)t6;

  using T7 = MergeTransformationStrategy<3>;
  T7 *t7 = nullptr;
  (void)t7;

  using T8 = CellwiseDistributor<REAL>;
  T8 *t8 = nullptr;
  (void)t8;

  using T9 = CellwiseDistributor<INT>;
  T9 *t9 = nullptr;
  (void)t9;

  // --- Vranic merging kernels ----------------------------------------------
  using V0 = VranicMergingKernels<2>;
  V0 *v0 = nullptr;
  (void)v0;

  using V1 = VranicMergingKernels<3>;
  V1 *v1 = nullptr;
  (void)v1;

  // --- Reaction data types -------------------------------------------------

  using D0 = FilteredMaxwellianSampler<2, ConstantRateCrossSection>;
  D0 *d0 = nullptr;
  (void)d0;

  using D1 = FilteredMaxwellianSampler<3, ConstantRateCrossSection>;
  D1 *d1 = nullptr;
  (void)d1;

  using D2 = SpecularReflectionData<2>;
  D2 *d2 = nullptr;
  (void)d2;

  using D3 = SpecularReflectionData<3>;
  D3 *d3 = nullptr;
  (void)d3;

  // --- Reaction kernel pre-requisites --------------------------------------

  using P0 = Properties<INT>;
  P0 *p0 = nullptr;
  (void)p0;

  using P1 = Properties<REAL>;
  P1 *p1 = nullptr;
  (void)p1;

  using P2 = ArgumentNameSet<INT>;
  P2 *p2 = nullptr;
  (void)p2;

  using P3 = ArgumentNameSet<REAL>;
  P3 *p3 = nullptr;
  (void)p3;

  std::cout << "consumer_smoke: linked against libVANTAGE-Reactions.so OK\n";
  return 0;
}
