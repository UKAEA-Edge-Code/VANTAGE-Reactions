**************************************
Building new reaction data and kernels
**************************************

Introduction
============

Two of the main focus points of Reactions are extensibility and portability. Unfortunately, these often clash, since special consideration is needed when working with SYCL.
For example, many low level objects in Reactions require both a host and device type to satisfy SYCL requirements (see SYCL specification for rules). This is the case for both reaction data and kernels. 

Another common point between reaction data and kernels is the need to handle required properties. This is usually done by defining a small namespace and by using that namespace in the
host type constructor to set the correct indices required for accessing the various NESO-Particles objects in :class:`ParticleLoop` calls. Examples will be given below. 

Reaction data
=============

When building new reaction data objects, the following should be kept in mind:

#. The device type should be constructed in the host type constructor and stored on the host type
#. Required properties need to be passed to the base reaction data class :class:`ReactionDataBase` in the form of :class:`Properties` containers - so they are best stored as static class members
#. Any required properties must be associated with public indices on the device type. Implement :code:`index_on_device_object()` and call it after constructing the device object; composition also calls this hook when property indices change.
#. The base class exposes the stored device object through :code:`get_on_device_obj()`.

Below is an example of how to build a set of reaction data objects, both host and device. The example is essentially a reproduction of the :class:`FixedCoefficientData` implementation. 

.. literalinclude:: ../example_sources/example_new_reaction_data.hpp
   :language: cpp
   :caption: Defining a new reaction data class on host and device

Property requirements and argument packs
----------------------------------------

Each host reaction-data object holds one current argument pack and a separate
property-name map. The pack is the authoritative layout for both loop arguments
and device property indices. Use :code:`get_arg_pack()` to inspect this layout and
the protected :code:`get_properties_map()` accessor to resolve property IDs to
names.

For single-particle data, :class:`ReactionDataBase` converts the supplied
:class:`Properties` containers into a :code:`SingleReactionDataArgumentPack`.
Regular and ephemeral requirements are merged into the same INT and REAL name
sets, with duplicate names removed. Pair data use a
:code:`PairReactionDataArgumentPack`, which keeps the INT and REAL requirements
of particles A and B separate.

Composition merges child packs and supplies the merged layout to every child.
Nested composites propagate that layout recursively before rebuilding their
device objects. A child's current pack therefore includes properties needed by
its siblings; it does not retain a separate narrow construction-time pack.

Implement :code:`index_on_device_object()` using the current pack's
:code:`find_index()` lookups. For enum-based requirements, resolve names through
:code:`get_properties_map()`; data taking explicit symbols can look up those
symbol names directly. Do not cache indices from the initial requirement set:
merging adds names to the sorted layout and can shift existing indices.
Constructor-only indexing is insufficient when the object is later composed.

Reaction kernels
================ 

Developing new kernels follows a similar process to the construction of new reaction data:

#. The device type should be stored on the host type and constructed in the host constructor in order to hide it from users 
#. Required properties work similarly to reaction data, but descendant particle properties are also required
#. Any required properties must be associated with public indices on the device type in the host constructor, as with reaction data 
#. A device object getter must be defined 
#. If descendant particles are produced, a call to :code:`set_descendant_matrix_spec<ndim_velocity,num_products_per_parent>()` is required. Here the template arguments are the velocity 
   space dimension and the number of products in the reaction 
#. When defining kernels that use data produced in :code:`DataCalculator` objects, the required data dimensionality must be set 

Below is a version of the charge exchange kernels with comments explaining the above points as well as othern nuances. 

.. literalinclude:: ../example_sources/example_cx_kernel_definition.hpp
   :language: cpp
   :caption: Defining CX kernels from scratch

.. _header_impl_split:
   
Declaration/Definition split (compiled build)
=============================================

Some headers under ``include/reactions_lib/`` are split into 
declarations (``X.hpp``) and definitions (``X.cpp`` which live in the ``src`` directory). 

When adding new non-templated objects or non-inline functions to the library:

#. put the declaration in ``X.hpp`` and the definition in ``X.cpp`` (in the ``src`` directory)
   (create the pair if adding a new component)
#. add the ``#include`` of the new ``X.hpp`` into ``include/reactions/reactions.hpp``.

This is the pattern for non-templated objects and non-inline functions. 
See :ref:`instantiating_unsupported_combos` for more details on instantiating new templated objects.
