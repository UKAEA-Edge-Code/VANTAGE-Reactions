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
#. Required properties need to be passed to the base reaction data class :class:`ReactionDataBase` in the form of :class:`Properties` containers - so they are best stored in a namespace outside of the host class
#. Any required properties must be associated with public indices on the device type - this has to be done in the host type constructor
#. The device type stored on the host type must be made accessible through the definition of a :code:`get_on_device_obj()` getter 

Below is an example of how to build a set of reaction data objects, both host and device. The example is essentially a reproduction of the :class:`FixedCoefficientData` implementation. 

.. literalinclude:: ../example_sources/example_new_reaction_data.hpp
   :language: cpp
   :caption: Defining a new reaction data class on host and device

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
