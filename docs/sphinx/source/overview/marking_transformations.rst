***************************
Marking and transformations
***************************

Introduction
============

A common use case when dealing with NESO-Particles :class:`ParticleGroup` objects is selecting particles based on some condition(s) and applying transformations to them.
For example, assuming a :class:`ParticleGroup` contains many different species of particles, one might want to select all particles of a given species that also have weight below some threshold, and then merge them to avoid tracking many small particles. 

In order to accommodate the above, VANTAGE-Reactions offers a uniform interface for :class:`MarkingStrategy`, :class:`TransformationStrategy`, and :class:`TransformationWrapper` objects.

Marking Strategies
==================

A marking strategy is the abstract wrapper class for the creation of NESO-Particles particle subgroups. The below strategies are the two currently implemented.

.. literalinclude:: ../example_sources/example_marking_strategy.hpp
   :language: cpp
   :caption: Example of several marking strategies

As demonstrated above, marking strategies are composed by applying them one after the other to get particle subgroups where particles respect all conditions.

Transformation Strategies
=========================

Once a suitable subgroup is constructed, we use :class:`TransformationStrategy` objects to apply any required transformation to those particles. All transformation strategies have the transform method which is applied to particle subgroups. Examples of the various built-in strategies are given below.

Particle Removal Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~

Often we wish to remove some particles from the simulation. For example, this might be due to their weight being too low to track. The following code will remove all particles with weight below a given threshold using marking and transformation strategies.

.. literalinclude:: ../example_sources/example_removal_strategy.hpp
   :language: cpp
   :caption: Example of using a removal strategy

ParticleDatZeroer
~~~~~~~~~~~~~~~~~

VANTAGE-Reactions assumes that particles carry information of their contribution to various sources that need to be projected onto the grid. A common requirement in these situations is to reset the values of sources on the particles after projection. The library offers the ParticleDatZeroer transformation strategy that allows the user to accomplish this.

.. literalinclude:: ../example_sources/example_zeroer_strategy.hpp
   :language: cpp
   :caption: Example of zeroing out real-valued particle data

Accumulator Strategies
~~~~~~~~~~~~~~~~~~~~~~

Another common requirement is the accumulation of particle properties cellwise. This is a requirement for finite volume methods (projection of sources) as well as general particle data analysis (weighted averages of quantities). Two classes of transformation strategies are provided for this use.

.. literalinclude:: ../example_sources/example_accumulator_strategy.hpp
   :language: cpp
   :caption: :class:`CellwiseAccumulator` and :class:`WeightedCellwiseAccumulator` example

Composite Strategy
~~~~~~~~~~~~~~~~~~

Sometimes multuple strategies need to be applied in order. It is possible to compose transformation strategies by adding them to a composite strategy, allowing all of them to be applied with one transform call. This is particularly useful in the construction of :class:`TransformationWrapper` objects (see below), where there is a hook left for a single transformation strategy.

.. literalinclude:: ../example_sources/example_composite_strategy.hpp
   :language: cpp
   :caption: Applying accumulator and zeroer strategies as part of a composite strategy

Particle Merging Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~

VANTAGE-Reaction implements a simplified merging algorithim from [VRANIC2015]_. It assumes that all particles being merged are of the same species (i.e. have the same mass) and that they are non-relativistic. 
In the original paper, the authors merge particles within momentum space cells, while we merge all particles in the subgroup passed to the transformation strategy (and we use the momentum space bounding box in 3D to determine the plane in which the merged particle momenta lie). 

Particles are merged cell-wise into 2 particles. The properties modified by the merging algorithm are the positions, weights, and momenta/velocities. Other properties are taken from 2 other particles in the passed subgroups, i.e. properties like cell and species IDs should be copied consistently, but all other properties should be considered undefined. As such, merging should only be invoked once all particle properties have been used for their respective purposes. 

The implementation of :class:`MergeTransformationStrategy` is available in 2D and 3D, and, given the above considerations, is easily used. 

.. literalinclude:: ../example_sources/example_merging_strategy.hpp
   :language: cpp
   :caption: An example of constructing a merging strategy in 2D

Transformation Wrappers
=======================

Often we wish to encapsulate both some marking conditions as well as the transformation we wish to perform into a single object. 
A common example is performing merging on multiple different species, but only on particles where weight is less than some threshold. In other words, the transformation we wish to perform is fixed, but only some of the marking conditions are fixed, while others vary. In this case the library offers the :class:`TransformationWrapper` class, which wraps marking conditions and a transformation strategy, and applies them to a :class:`ParticleGroup`. We can then fix the instruction "Merge all particles with weight < threshold", and can extend it with other conditions, such as "and with species ID = 1". 

.. literalinclude:: ../example_sources/example_transformation_wrapper.hpp
   :language: cpp
   :caption: An example of a :class:`TransformationWrapper` being used to remove particles with low weights for two different ID values

.. [VRANIC2015] Vranic et al. - Particle merging algorithm for PIC codes https://www.sciencedirect.com/science/article/pii/S0010465515000405

