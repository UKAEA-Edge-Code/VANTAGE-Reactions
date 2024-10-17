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

Both data and kernels, in executing their responsibilities, access particle data, and use the property map system [TODO: add before Reactions section]

Reaction data
=============



