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

#include <reactions/reactions.hpp>

#include <iostream>

int main() {
  using namespace VANTAGE::Reactions;

  // Name a library-shipped instantiation. ODR-using the type (even via a
  // pointer) requires the class template to be complete, which forces the
  // consumer TU to respect the extern-template declaration and let the library
  // provide the definition. If the library did not ship this instantiation the
  // link would fail with an unresolved symbol at the point of ODR-use; the
  // extern-template decl keeps that ODR-use from emitting the symbol here.
  using LibraryShippedReaction =
      LinearReactionBase<1, FixedRateData, CXReactionKernels<2>,
                         DataCalculator<FixedRateData, FixedRateData>>;
  LibraryShippedReaction *p = nullptr;
  (void)p;

  std::cout << "consumer_smoke: linked against libVANTAGE-Reactions.so OK\n";
  return 0;
}
