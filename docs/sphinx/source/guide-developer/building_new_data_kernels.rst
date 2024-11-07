**************************************
Building new reaction data and kernels
**************************************

Introduction
============

Two of the main focus points of Reactions are extensibility and portability. Unfortunately, these often clash, since special consideration is needed when working with SYCL.
For example, many low level objects in Reactions require both a host and device type to satisfy SYCL requirements. This is the case for both reaction data and kernels. 

Another common point between reaction data and kernels is the need to handle required properties. This is usually done by defining a small namespace and by using that namespace in the
host type constructor to set the correct indices required for accessing the various NESO-Particles objects in :class:`ParticleLoop` calls. Examples will be given below. 

Reaction data
=============

When building new reaction data objects, the following should be kept in mind:

#. The device type should be constructed in the host type constructor and stored on the host typ 
#. Required properties need to be passed to the base reaction data class :class:`ReactionDataBase` in the form of :class:`Properties` containers - so they are best stored in a namespace outside of the host class
#. Any required properties must be associated with public indices on the device type - this has to be done in the host type constructor
#. The device type stored on the host type must be made accessible through the definition of a :code:`get_on_device_obj()` getter 

Below is an example of how to build a set of reaction data objects, both host and device. The example is essentially a reproduction of the :class:`FixedCoefficientData` implementation. 

.. literalinclude:: ../example_sources/example_new_reaction_data.hpp
   :language: cpp
   :caption: Defining a new reaction data class on host and device
