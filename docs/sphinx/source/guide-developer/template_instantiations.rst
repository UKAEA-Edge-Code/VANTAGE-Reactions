******************************
Runtime instantiantion details
******************************

All shipped instantiations are compiled into a **single translation unit**
(``src/instantiations/instantiations.cpp``). The non-template definitions live in a second, 
lightweight TU (``src/compiled/definitions.cpp``) — see the guide for
the header/impl split behind it: :ref:`header_impl_split`.

Baseline, not a closed surface
==============================

The shipped set is a **pay-once baseline** for select configurations
(e.g the 2D/3D kernels baseline, transformation strategies), pre-compiled into the ``.so`` and suppressed in consumers via
``extern template``. It is *not* a closed enumeration of every
``ReactionData x ReactionKernels x DataCalculator`` combination — the library's open-combinatorial
design intentionally pushes non-baseline compositions onto the consumer, which
compiles them from headers exactly as in header-only mode.

.. _instantiating_unsupported_combos:

Instantiating unsupported combinations
======================================

The library only ships with a few combinations but every other combination is still
**open**, if they're needed then: include the public headers, and rely on implicit
instantiation in the consumer TU.

Alternatively, add your own ``template class …``
explicit instantiation in a dedicated ``.cpp`` and pair it with
an ``extern template class …`` declaration in the consumer's own header
(the library does the same thing in ``extern_templates.hpp``). 

An example of this is in the ``test/unit`` directory, where there's and ``test_extern_templates.hpp`` in the ``test/unit/include`` directory 
and a ``test_instantiations.cpp`` in the ``test/unit/instantiations`` directory. These include the instantiations needed for the unit tests.

Adding to this framework is really only feasible if at least a few of the template instantiations of the templated objects are known. 
For example, this would not be fully applicable if using template parameter packs, where only a subset of
all possible combinations in that case might be known and could be added. 
In that case, it's recommended to simply lean on implicit instantiation in the consumer TUs that may use it.

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
To test it, install VANTAGE-Reactions via :code:`spack install && spack load vantagereactions`, then from within :code:`test/external_consumer` configure and build it via :code:`cmake . && make`. Run it with :code:`./consumer_smoke`.
