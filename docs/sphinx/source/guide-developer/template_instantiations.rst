******************************
Runtime instantiantion details
******************************

All shipped instantiations are compiled into a **single translation unit**
(``src/instantiations/instantiations.cpp``); this is intentional
and keeps the library's compile-time memory footprint characterised by one
(heavy) TU rather than spread across many. The non-template definitions that the
instantiations need live in a second, lightweight TU
(``src/compiled/definitions.cpp``) — see the guide for
the header/impl split behind it: :ref:`header_impl_split`.

Baseline, not a closed surface
==============================

The shipped set is a **pay-once baseline** for select configurations
(the example reactions, the AMJUEL ionisation composition, and the 2D/3D
kernel baseline), pre-compiled into the ``.so`` and suppressed in consumers via
``extern template``. It is *not* a closed enumeration of every
``ReactionData x ReactionKernels x DataCalculator`` combination — the library's open-combinatorial
design intentionally pushes non-baseline compositions onto the consumer, which
compiles them from headers exactly as in header-only mode.

Instantiating unsupported combinations
======================================

The library ships a few combinations because they exercise the common
example and derived reactions, the heavy kernel/data types, and the most
frequently used transformation strategies. Every other combination is still
**open**: include the public headers, then either rely on implicit
instantiation in the consumer TU or add your own ``template class …``
explicit instantiation in a dedicated ``.cpp`` and (optionally) pair it with
an ``extern template class …`` declaration in the consumer's own header
(the library does the same thing in ``extern_templates.hpp``).

If any new templated objects are added to the library, their explicit instantiation can be added to ``instantiations.cpp`` and its ``extern template`` to ``extern_templates.hpp``. 
This is really only feasible if at least a few of the template instantiations of the templated objects are known. 
For example, this would not be fully applicable if using template parameter packs, only a subset of
all possible combinations in that case might be known and could be added. In this case, it's recommended to simply lean on implicit instantiation in the consumer TUs that may use it.

Verifying the installed library
===============================

A standalone CMake project lives at ``test/external_consumer/`` that
configures against only the installed tree via
``find_package(VANTAGE-Reactions)``, links the installed library, and
ODR-uses (via a null pointer) **every** shipped instantiation listed
above. Each ODR-use forces the consumer TU to respect the matching
``extern template`` declaration and let the library provide the
definition, so a missing instantiation surfaces as an unresolved symbol
at link time. It is intended to be run as a post-install smoke test to confirm the runtime library is genuinely
linkable without the source tree.. 
To test it, install VANTAGE-Reactions via `spack install && spack load vantagereactions`, then from within ``test/external_consumer`` configure and build it via `cmake . && make`. Run it with `./consumer_smoke`.
