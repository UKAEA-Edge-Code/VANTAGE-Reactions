******************************
Runtime instantiantion details
******************************

All shipped instantiations are compiled into translation units listed in the 
``src`` directory.

Baseline only
=============

The shipped set is a **pay-once baseline** for select configurations
(e.g the 2D/3D kernels baseline, transformation strategies), pre-compiled into the ``.so`` and suppressed in consumers via
``extern template``. It is *not* a closed enumeration of every
``ReactionData x ReactionKernels x DataCalculator`` combination — the library's open-combinatorial
design intentionally pushes non-baseline compositions onto the consumer, which
compiles them from headers.

.. _instantiating_unsupported_combos:

Instantiating unsupported combinations
======================================

The library only ships with a few combinations but every other combination is still
**open**. 

If they're needed then: include the public headers (via ``#include <reactions/reactions.hpp>``), and rely on implicit
instantiation in the consumer TU.

Alternatively, after defining a template struct/class in a ``.hpp`` file, add an ``extern template class …`` declaration to it, add the corresponding ``template class …``
explicit instantiation in a dedicated ``.cpp``.

In the public headers, an example would be, ``vranic_merging_kernels.hpp`` which includes a template struct definition and ``extern template class`` declarations. 
The corresponding explicit template instantiations live in ``vranic_merging_kernels.cpp``.

Adding to this framework is really only feasible if at least a few of the template instantiations of the templated objects are known. 
For example, this would not be fully applicable if using template parameter packs, where only a subset of
all possible combinations in that case might be known and could be added. 
In that case, it's recommended to simply lean on implicit instantiation in the consumer TUs that may use it.

Verifying the installed library
===============================

A standalone CMake project lives at ``test/external_consumer/`` that
configures against only the installed tree via
``find_package(VANTAGE-Reactions)``, links the installed library, and
ODR-uses (via a null pointer) **every** shipped template instantiation. 
Therefore if any new instantiations are added to the shipped set then be sure to add their usage to ``test/external_consumer/consumer_smoke.cpp``.

Each ODR-use forces the consumer TU to respect the matching
``extern template`` declaration and let the library provide the
definition, so a missing instantiation surfaces as an unresolved symbol
at link time. It is intended to be run as a post-install smoke test to confirm the runtime library is genuinely
linkable without the source tree.
To test it, install and load VANTAGE-Reactions via :code:`spack install && spack load vantagereactions`, 
then from the repo directory configure and build it via 

::

  cmake -S test/external_consumer -B build-consumer -DCMAKE_BUILD_TYPE=RelWithDebInfo
  cmake --build build-consumer
  ./build-consumer/consumer_smoke
