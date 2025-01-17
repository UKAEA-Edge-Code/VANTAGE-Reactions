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

