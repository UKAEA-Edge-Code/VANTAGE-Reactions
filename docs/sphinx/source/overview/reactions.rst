******************************
Reactions and their components
******************************

What reactions (the abstraction) are
====================================

We refer to reactions as any process that involves one or more ingoing particles (physical or otherwise) interacting with other particles in the simulation or fields (stored as ParticleDats), as well as one or more of the following:

#. The production of new particles in the simulation (with their own velocities, weights, and various internal states)
#. The modification of ingoing particle properties (weights, etc.)
#. Feedback on fields (such as particle/energy sources - assumed stored on the ingoing particle and projected onto the mesh separately)

In the abstract, a reaction is fully defined by:

#. Ingoing and outgoing particle IDs (these are notionally unique integer labels associated with particle species)
#. Any (per particle) data and associated calculation methods needed to apply the reaction - most notably the reaction rate 
#. How the properties (NESO-Particles ParticleDats) of the parents and children are modified/generated

We make a distinction between linear and non-linear reactions in the particle sense. A linear reaction is any reaction where only one of the reactants is represented as a particle.
There are no constraints on the number of reaction products in linear reactions. In contrast, a non-linear reaction is a reaction where two or more reactants are represented as particles in the simulation.
An example of a non-linear reaction would be an elastic collision between two neutrals of the same species. There exist linearisation techniques for some of these reactions, so the initial focus of the framework is linear reactions.

Linear Reaction structure
=========================

The main components of reactions are the reaction data and reaction kernel objects. Their overall responsibilities are as follows:

* Reaction data - calculate the per particle data required for the application of the reaction. This could be reaction rates, values randomly sampled from some distributions, etc. 
* Reaction kernels - define the properties of the products of the reaction (velocities, weights, internal states), as well as the feedback on fields and the parent particle 

The key idea behind this separation of concern is the ability to separate the data and the physics, and allow the combination of different data calculation methods and different reaction physics. For example, the physics of an ionisation reaction is the same regardless of the reaction rate or the energy cost of the reaction, and the goal of flexibility in Reactions has lead to the data+kernel design. 

The implementation of reactions, as well as reaction kernels will be covered in the developer guide, as it involves considerations of SYCL host and device types, as well as NESO-Particles :class:`ParticleLoop` constructs. 

Both data and kernels, in executing their responsibilities, access particle data, and use the property map system.

Reaction data and kernels are invoked in the two main :class:`ParticleLoop`-containing methods in the linear reaction class, with the idea that data is calculated first, storing anything needed for the application of the kernels or for the global management of reaction application. Both loops are assumed to be invoked cell-wise, which allows for the reuse of various buffers.

Reaction data and the LinearReaction data loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reaction data objects calculate a fixed number of components per particle. For example, data objects used to calculate reaction rates have a single component, while an object that is sampling velocities from a distribution might have two or more components. Each reaction needs at least one reaction data object - responsible for the calculation of the rate for the reaction. Further data calculation can be bundled using the :class:`DataCalculator` container of reaction data. 

Invoking the rate loop on a reaction object does the following:

#. Calculates the reaction rate and stores it in a local buffer used to apply the reaction using the kernels
#. Adds the calculated reaction rate to a total reaction rate :class:`ParticleDat` - used in the global management of reaction application
#. Calculates and stores any :class:`DataCalculator` results for use by the kernels

Reaction kernels and the product loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With any data required to apply the reaction calculated and stored for some particles, the next step is to apply the reaction, which might involve feedback on fields and the ingoing particle, as well as some specification of product properties following the reaction. This is (semi-)independently specified by choosing a reaction kernel. The only requirement on the data that a kernel might have is that any required data exists, i.e. that the total dimensionality of data conforms with whatever the kernel requires. For example, if a kernel requires 2 sampled velocities, the :class:`DataCalculator` must produce a total of 2 data values per perticle. Other than this requirement, data and kernels are independent. 

Each kernel object consists of four kernel functions, in order to allow for extensibility. These are:

#. The scattering kernel - nominally specifies the velocities of the products
#. The weight kernel - nominally specifies the weights of the products
#. The transformation kernel - nominally specifies any complex internal state changes of the product
#. The feedback kernel - nominally specifies any feedback on the parent particle an on any fields, such as fluid sources

The above are applied in that order, and the product loop stores any products/children into a separate particle group in order to allow for any transformations before they are added into the group with the parents (see :class:`ReactionController` documentation). 

As noted above, both kernels and data specify their required particle properties using the property enum+map system, so that on-the-fly remaping of required variables is possible. 

Putting a linear reaction together
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As noted above, to construct a linear reaction, we need to know the state IDs of the ingoing (parent/reactant) and outgoing (children/products) particles, as well as the data and kernels. 

Below an example with the built-in charge-exchange kernels using fixed values for all of the data. It demonstrates the pipeline needed to build a linear reaction object, as well as some of the method calls on the object relating to the two loops described above. More detailes on the individual data and kernel objects and their required properties will be presented below. 


.. literalinclude:: ../example_sources/example_linear_reaction_CX.hpp
   :language: cpp
   :caption: Constructing a CX reaction with a fixed rate and with a beam of ions


