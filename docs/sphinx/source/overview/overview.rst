********
Overview
********

.. _Introduction:

Introduction
============ 

VANTAGE-Reactions is part of the scalable, flexible, and extensible VANTAGE neutral particle modelling suite for the tokamak plasma exhaust. It is built on top of the NESO-Particles library with the goal to handle (spatially) 0D particle transformations, in particular those due to collisional/reactive processes, including surface-related reactions. 

VANTAGE-Reactions is structured as a library, providing features necessary for treating processes of interest while relying on other components for coupling, geometry, and most post-processing and analysis.

Feature overview
~~~~~~~~~~~~~~~~

Features provided by the library are:

#. An extensible reaction abstraction, designed to be modular, separating the data and the actions on the parents/products 
#. A uniform and extensible interface for producing particle subgroups by composing marking strategies 
#. A uniform and extensible interface for defining, composing, and applying transformations to particle groups, including marking 
#. An interface for the simultaneous application of multiple reactions to the relevant particles and the handling of reaction products
#. Various helper interfaces for defining particle species as well as generating uniform particle specs 
#. A collection of pre-built reactions/reaction data/reaction kernels

VANTAGE particle model
~~~~~~~~~~~~~~~~~~~~~~

VANTAGE uses a weighted particle model, where particles are described by their positions, velocities, and weights :math:`w`, with those weights representing a number of physical particles contained with the macro particle. The weights are assumed to be variable. 

Particles can also hold other properties, depending on the use case:

* Additional state-specifying data (e.g. species index, internal state specification, etc.)
* Values of various fields evaluated at particle locations (e.g. plasma densities, temperatures, etc.) - **NOTE**: VANTAGE-Reactions does not provide utilities for evaluation and projection of particle properties, which depend on the user's application and mesh choices
* Source terms/values to be projected from the particles onto the fields (e.g. particle, momentum, energy sources due to reactions)

In order to construct a NESO-Particles :class:`ParticleGroup` compatible with the requirements of the above particle model, VANTAGE-Reactions offers utilities for the specification of :class:`Species` and various particle properties, which are then used in the reaction model.

VANTAGE-Reactions reaction model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The central concept of the library is that of a reaction, where some number of ingoing particles react/collide, and produce a set of products. As an example, consider the following abstract reaction

 .. math:: 
   
   A + B = 2C + D

where A and B are the reactants (ingoing species) and C and D are the products (outgoing species). In general, one could treat all of the reactants (and products) of a reaction as particles. However, a simplifying assumption is that only one reactant species is treated as a particle, whereas all others are represented through some fields, with which the particle interacts. We refer to these as **linear reactions**. For example, electron-impact ionisation of an atom can be treated as a single particle interacting with an electron fluid. 

Each **linear reaction** is uniquely defined by its reactants and products, as well as how any reaction data (such as rates) are calculated, how the reaction modifies properties of the parent (including any sources stored on the parent that need to be projected onto various fields), and how it sets the properties of the products. This is explained in detail further in the documentation, but we cover some basic points here. 

As noted above, the particle model in VANTAGE assumes that particles have some weight :math:`w`. In reactions, some or all of this weight participates. While there are other modes available, the default mode assumes that some weight of the particle, defined by the value of a reaction rate :math:`K` (which could be a function of any particle properties), is reduced according to

 .. math::

    \frac{dw}{dt} = -K

which is, by default, solved using an Euler step. Thus, the weight participating in the reaction with rate :math:`K` during a step of length :math:`\Delta t` is :math:`K \Delta t`. This weight is then accounted for in the products of the reaction (whether represented as particles or fields). 

However, it is not too difficult to see that, if :math:`K` is a function of :math:`w` (as it usually is), naively applying two reactions in succession is going to produce different results depending on the ordering. To address this, VANTAGE-Reactions defines a controller class that handles the simultaneous (self-consistent) application of reactions and the addition of products. 

Due to allowing such schemes as the above, i.e. converting some particle weight into products at every step, there is no natural protection from an explosion in the number of particles (sometimes referred to as collisional cascades). To address this, as well as many other problems requiring the modification of particle groups and subgroups outside of individual reaction application, VANTAGE-Reactions defines abstractions for particle selection and transformation. Examples include selecting and merging low weight particles or accumulating various sources cellwise. Most importantly, these transformations can then be hooked into the controller abstraction, allowing for the definition of complex reaction application pipelines.

Using VANTAGE-Reactions
=======================

VANTAGE-Reactions is meant to be single component of a modelling suite. It does not provide the following:

#. Mesh utilities (it is only aware of NESO-Particles cells)
#. Projection and evaluation of quantities on the mesh
#. Particle advection and boundary intersection detection 
#. A standard set of reactions and reaction data (but does provide the utilities to define them)  

The library is fully agnostic to the choice of mesh (though it does provide some simple routines that can be used in the finite volume case), as well as the choice of particle advection method. 

To use the library, the user needs to:

#. Define species objects containing basic information about the particle species in the simulation
#. Define a compatible NESO-Particles :class:`ParticleGroup`, for which VANTAGE-Reactions provides helper function. 
#. Define and add individual reaction objects to the controller object 
#. Define and set the transformations they wish to apply on reactants and products in addition to the reactions (such as low-weight particle merging)
#. Use the controller to apply an Euler step in their integration scheme where the desired reactions and transformations will be applied to the :class:`ParticleGroup` 

All of the above steps are individually documented in the library documentation, with examples where appropriate.

