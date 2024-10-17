********
Overview
********

What Reactions (the library) is
===============================

Reactions is a scalable, flexible, and extensible library for adding reactions/particle transformations to particle codes built on top of the NESO-Particles library.

Features provided by the library are:

#. An extensible reaction abstraction, designed to be modular, separating the data and the actions on the parents/products 
#. A uniform and extensible interface for producing particle subgroups by composing marking strategies 
#. A uniform and extensible interface for defining, composing, and applying transformations to particle groups, including marking 
#. An interface for the simultaneous application of multiple reactions to the relevant particles and the handling of reaction products
#. Various helper interfaces for defining particle species as well as generating uniform particle specs 
#. A collection of pre-built reactions/reaction data/reaction kernels

What Reactions isn't
====================

This library deals only with the definition and application of particle transformations useful when dealing with reacting particles in particle codes. It is **NOT** a particle code itself. 
It does not deal with moving the particles around, and is mesh agnostic. It does not define a standard set of reactions/species but provides the tools to do so. 

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
