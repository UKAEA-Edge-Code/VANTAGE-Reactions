********
Overview
********

.. _Introduction:

Introduction
============ 

VANTAGE-Reactions is part of the scalable, flexible, and extensible VANTAGE neutral particle modelling suite for the tokamak plasma exhaust. It is built on top of the NESO-Particles library with the goal to handle (spatially) 0D particle transformations, in particular those due to collisional/reactive processes. 

VANTAGE-Reactions is structured as a library, providing features necessary for treating processes of interest (see below for a list of features). The main concept is that of a reaction, where some number of ingoing particles react/collide, and produce a set of products. As an example, consider the following abstract reaction

.. math:: 
   
   A + B = 2C + D

where A and B are the reactants (ingoing species) and C and D are the products (outgoing species). In general, one could treat all of the reactants (and products) of a reaction as particles. However, a simplifying assumption is that only one reactant species is treated as a particle, whereas all others are represented through some fields, with which the particle interacts. For example, electron-impact ionisation of an atom can be treated as a single particle interacting with an electron fluid. Such **linearised reactions** can then be applied on each particle using NESO-Particles :class:`ParticleLoop` features. Each linear reaction is uniquely defined by its reactants and products, as well as how any reaction data (such as rates) are calculated, how the reaction modifies properties of the parent, and how it sets the properties of the products. This is explained in detail further in the documentation.

In VANTAGE-Reactions, we assume that particles have some weight :math:`w` and it is this weight, together with other particle properties (velocities, internal states, etc.) that are manipulated when applying individual reactions. For example, for a single reaction (in the default Euler step), the change in weight associated with it is :math:`K \Delta t`, where :math:`K` is the reaction rate and :math:`\Delta t` is the time step. It is not too difficult to see that, if :math:`K` is a function of :math:`w` (as it usually is), naively applying two reactions in succession is going to produce different results depending on the ordering. To address this, VANTAGE-Reactions defines a controller class that handles the simultaneous (self-consistent) application of reactions and the addition of products. 

In a realistic tokamak exhaust simulation, one might encounter many different species of particles, with many different reactions, all producing many products. If products are naively added, collisional cascade is almost guaranteed, with many low-weight particles wasting computational effort. VANTAGE-Reactions defines abstractions that can be used to define transformations on subgroups of particles, such as merging many low-weight particles, or that can be used to accumulate verious sources associated with reactions, such as plasma sources due to ionisation. These transformations can then be hooked into the controller abstraction to apply them in the reaction application pipeline. 

Furthermore, with the many different species present, VANTAGE-Reactions offers utilities for dealing with the construction of NESO-Particles :class:`ParticleSpec` objects compatible with its features.

Feature overview
================

Features provided by the library are:

#. An extensible reaction abstraction, designed to be modular, separating the data and the actions on the parents/products 
#. A uniform and extensible interface for producing particle subgroups by composing marking strategies 
#. A uniform and extensible interface for defining, composing, and applying transformations to particle groups, including marking 
#. An interface for the simultaneous application of multiple reactions to the relevant particles and the handling of reaction products
#. Various helper interfaces for defining particle species as well as generating uniform particle specs 
#. A collection of pre-built reactions/reaction data/reaction kernels


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

