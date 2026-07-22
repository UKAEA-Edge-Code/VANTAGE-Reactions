*********************************
Supported runtime instantiations
*********************************

By default ``VANTAGE-Reactions`` is built as a **compiled runtime library**
(SHARED, or STATIC when ``BUILD_SHARED_LIBS`` is OFF) and ships a curated set
of pre-compiled template instantiations and SYCL device code in
``libVANTAGE-Reactions.so``. Consumers link the installed library via
``find_package(VANTAGE-Reactions)`` instead of re-compiling the headers for
every supported reaction/data/kernel combination.

A header-only ``INTERFACE`` target is still available as an opt-out via
``-DVANTAGE_REACTIONS_HEADER_ONLY=ON``. In that mode every consumer compiles
the headers itself, exactly as before; there is no compiled ``.so`` to link.

All shipped instantiations are compiled into a **single translation unit**
(``src/reactions_lib/instantiations/instantiations.cpp``); this is intentional
and keeps the library's compile-time memory footprint characterised by one TU
rather than spread across many. If a future memory benchmark shows regression,
the TU will be split rather than coverage dropped.

Two names refer to the same exported target:

* the in-source build alias ``VANTAGE::Reactions``,
* the installed EXPORT target ``VANTAGE-Reactions::VANTAGE-Reactions``.

Link either one to a consumer target; consumers are recommended to put them
in ``target_link_libraries(<consumer> PRIVATE <name>)``.

Supported instantiations
========================

The instantiations below are compiled into the library and are visible to
every consumer without further work. The matching ``template class …``
lines live in ``src/reactions_lib/instantiations/instantiations.cpp``; the
``extern template class …`` declarations that suppress re-instantiation in
consumer TUs live in ``src/reactions_lib/extern_templates.hpp`` (included
from the public ``reactions.hpp``).

Reactions used in the examples
------------------------------

* ``LinearReactionBase<1, FixedRateData, CXReactionKernels<2>,
  DataCalculator<FixedRateData, FixedRateData>>``
* ``LinearReactionBase<0, FixedRateData, IoniseReactionKernels<2>,
  DataCalculator<FixedRateData>>``
* ``LinearReactionBase<1, FixedRateData, RecombReactionKernels<2, 2>,
  DataCalculator<FixedRateData, FixedRateData, FixedRateData>>``
* ``LinearReactionBase<1, FixedRateData, LinearScatteringKernels<2, true>,
  ScatteringDataCalculator>``
* ``LinearReactionBase<1, FixedRateData, CXReactionKernels<3>,
  DataCalculator<FixedRateData, FixedRateData, FixedRateData>>``
* ``LinearReactionBase<1, FixedRateData, LinearScatteringKernels<3, true>,
  ScatteringDataCalculator3D>`` (3D counterpart of the 2D scattering reaction;
  ``ScatteringDataCalculator3D`` is the 3D analogue of
  ``ScatteringDataCalculator``, built from ``SpecularReflectionData<3>``)

Derived reactions
-----------------

* ``ElectronImpactIonisation<FixedRateData, FixedRateData, 2>``
* ``ElectronImpactIonisation<AMJUEL1DData<9>, FixedRateData, 2>``
  (the AMJUEL-backed ionisation composition used by the ``example_amjuel*``
  snippets)
* ``Recombination<FixedRateData, DataCalculator<FixedRateData,
  FixedRateData, FixedRateData>, 2>``

Kernel classes
--------------

* ``CXReactionKernels<2>``
* ``IoniseReactionKernels<2>``
* ``RecombReactionKernels<2, 2>``
* ``LinearScatteringKernels<2, true>``
* ``CXReactionKernels<3>``
* ``LinearScatteringKernels<3, true>``

:class:`DataCalculator` specialisations
------------------------------------------

* ``DataCalculator<FixedRateData>``
* ``DataCalculator<FixedRateData, FixedRateData>``
* ``DataCalculator<FixedRateData, FixedRateData, FixedRateData>``
* ``DataCalculator<VelocityReflectionPipeline>``

Common transformation strategies
--------------------------------

* ``CellwiseAccumulator<REAL>``
* ``WeightedCellwiseAccumulator<REAL>``
* ``ParticleDatZeroer<REAL>``
* ``MergeTransformationStrategy<2>``

Downsampling strategies
-----------------------

* ``VranicMergingKernels<2>``
* ``DownsamplingStrategy<VranicMergingKernels<2>>``
* ``DownsamplingStrategy<SimpleThinningKernels>``

Other heavy reaction data types
-------------------------------

* ``FilteredMaxwellianSampler<2, ConstantRateCrossSection>``
* ``CellwiseReactionDataAccumulator<KinEnergyData>``

Baseline, not a closed surface
==============================

The shipped set is a **pay-once baseline** for the documented configurations
(the example reactions, the AMJUEL ionisation composition, and the 2D/3D
kernel baseline), pre-compiled into the ``.so`` and suppressed in consumers via
``extern template``. It is *not* a closed enumeration of every
``Data x Kernels x DataCalc`` combination — the library's open-combinatorial
design intentionally pushes non-baseline compositions onto the consumer, which
compiles them from headers exactly as in header-only mode. Check the list
above: if your reaction is listed, it is pre-covered; if not, instantiate it
yourself (see the next section).

Instantiating unsupported combinations
======================================

The library ships the combinations above because they exercise the common
example and derived reactions, the heavy kernel/data types, and the most
frequently used transformation strategies. Every other combination is still
**open**: include the public headers, then either rely on implicit
instantiation in the consumer TU or add your own ``template class …``
explicit instantiation in a dedicated ``.cpp`` and (optionally) pair it with
an ``extern template class …`` declaration in the consumer's own header
(the library does the same thing in ``extern_templates.hpp``).

In particular:

* reactions that use a ``ReactionData`` type other than ``FixedRateData`` are
  *not* in the shipped set *except* for ``ElectronImpactIonisation<AMJUEL1DData<9>, FixedRateData, 2>``
  (the documented AMJUEL ionisation composition); other AMJUEL degrees (e.g.
  ``AMJUEL1DData<N>`` with ``N != 9``) and other AMJUEL-backed reactions are not
  shipped — instantiate them yourself;
* recombinations with a ``DataCalculator`` arity not listed above (4+, or
  reordered types) are *not* shipped — instantiate them yourself;
* any new public reaction added to the library should also add its explicit
  instantiation to ``instantiations.cpp`` and its ``extern template`` to
  ``extern_templates.hpp`` (see ``src/reactions_lib/add_new_reactions.md``).

Verifying the installed library
===============================

A standalone CMake project lives at ``test/external_consumer/`` that
configures against only the installed tree via
``find_package(VANTAGE-Reactions)``, links the installed library, and
ODR-uses (via a null pointer) **every** shipped instantiation listed
above. Each ODR-use forces the consumer TU to respect the matching
``extern template`` declaration and let the library provide the
definition, so a missing instantiation surfaces as an unresolved symbol
at link time. It is intended to be run as a post-install smoke test
(`cmake --install`, then configure and build the consumer project
against the install prefix) to confirm the runtime library is genuinely
linkable without the source tree.
