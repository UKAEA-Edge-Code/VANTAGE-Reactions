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

A marking strategy is the abstract wrapper class for the creation of NESO-Particles particle subgroups. The main marking strategy is the :class:`MarkingStrategyDirect`, which is effectively a closure for the NESO-Particles subgroup constructor (see below how this enables transformation wrappers).

.. literalinclude:: ../example_sources/example_marking_strategy.hpp
   :language: cpp
   :caption: Example of the direct marking strategy


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

Another common requirement is the accumulation of particle properties cellwise. This is a requirement for finite volume methods (projection of sources) as well as general particle data analysis (weighted averages of quantities). Three classes of transformation strategies are provided for this use:

* `CellwiseAccumulator` - accumulating one or more properties cellwise
* `WeightedCellwiseAccumulator` - accumulating one or more properties cellwise while weighing them with the particle weights
* `CellwiseReactionDataAccumulator` - accumulating the result of a reaction data object, for use in cases when the first two are two restrictive

.. literalinclude:: ../example_sources/example_accumulator_strategy.hpp
   :language: cpp
   :caption: :class:`CellwiseAccumulator` and :class:`WeightedCellwiseAccumulator` example
   
.. literalinclude:: ../example_sources/example_reaction_data_accumulator_strategy.hpp
   :language: cpp
   :caption: :class:`CellwiseReactionDataAccumulator` example

Cellwise distributor strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similar to the requirement for accumulating properties cellwise, there are situations where we want to broadcast one or more property onto all particles in a cell. The `CellwiseDistributor` transformation strategy offers this, working like the inverse of the `CellwiseAccumulator`.


.. literalinclude:: ../example_sources/example_cellwise_distributor_strategy.hpp
   :language: cpp
   :caption: :class:`CellwiseDistributor` example

Composite Strategy
~~~~~~~~~~~~~~~~~~

Sometimes multuple strategies need to be applied in order. It is possible to compose transformation strategies by adding them to a composite strategy, allowing all of them to be applied with one transform call. This is particularly useful in the construction of :class:`TransformationWrapper` objects (see below), where there is a hook left for a single transformation strategy.

.. literalinclude:: ../example_sources/example_composite_strategy.hpp
   :language: cpp
   :caption: Applying accumulator and zeroer strategies as part of a composite strategy

Downsampling Strategies
~~~~~~~~~~~~~~~~~~~~~~~

A common problem in weighted particle methods is ensemble management, and in particular downsampling. VANTAGE-Reactions offers a framework for building downsampling transformation strategies, with a small number of them supplied through helper functions.

In general, downsampling strategies consist of the following steps:

1. Reduction - where a number of quantities across the particle ensemble are reduced, i.e. moments or other quantities are calculated
2. Downsampling - where the properties of the particles are modified in such a way that some of them are effectively marked for removal, while the remaining particles are modified according to the downsampling algorithm.
3. Removal - particles effectively "marked" for removal are removed 

The above can be applied on multiple downsampling groups separately, and the downsampling strategies that assume grouping expect that it has been prepared beforehand, by default using the `grouping_index` integer property. See uniform velocity binning below as an example of a binning transform.

Vranic Merging Strategy
^^^^^^^^^^^^^^^^^^^^^^^

VANTAGE-Reactions implements a version of the merging algorithim from [VRANIC2015]_. It assumes that all particles being merged are of the same species (i.e. have the same mass) and that they are non-relativistic. 
In the original paper, the authors merge particles within momentum space cells, while we merge all particles in the downsampling group, which could be a momentum/velocity space cell, but doesn't have to be. Correspondingly, in 3D we use the momentum space bounding box in 3D to determine the plane in which the merged particle momenta lie. 

Particles are merged cell-wise and downsampling-group-wise into 2 particles. The properties modified by the merging algorithm are the weights and momenta/velocities. Other properties are taken from 2 other particles in the passed subgroups, i.e. properties like cell and species IDs should be copied consistently, but all other properties should be considered undefined. As such, merging should only be invoked once all particle properties have been used for their respective purposes, such as recording sources. Note that the above means that the first two particles' positions in the downsampling group will be used as the merged particle positions.

.. literalinclude:: ../example_sources/example_vranic_merging_strategy.hpp
   :language: cpp
   :caption: An example of constructing the above merging strategy in 2D 

Simple Thinning Strategy
^^^^^^^^^^^^^^^^^^^^^^^^

A classic alternative to merging is particle thinning, i.e. removing some particles randomly while modifying the properties of the rest. The simple thinning strategy keeps particles with some probability - the `thinning_ratio`, scaling their weights with the inverse of that probability, while removing the rest of the particles. This procedure conserves particle weight on average only. 

.. literalinclude:: ../example_sources/example_simple_thinning_strategy.hpp
   :language: cpp
   :caption: An example of constructing a simple thinning strategy

**DEPRECATED** Legacy Merging Strategy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a purely cell-wise version of the above Vranic merging strategy, which also merges particles into the centre of mass. Both of these are drawbacks and the transformation is likely to be removed in the future. 

The implementation of of the legacy :class:`MergeTransformationStrategy` is available in 2D and 3D:

.. literalinclude:: ../example_sources/example_merging_strategy.hpp
   :language: cpp
   :caption: An example of constructing a merging strategy in 2D

Direct transformation strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In cases where the user wants to apply a custom lambda function to a particle subgroup or call a particular :class:`ParticleLoop` as a transformation strategy VANTAGE-Reactions supplies :class:`TransformationStrategyDirect` and :class:`TransformationStrategyLambda`.

.. literalinclude:: ../example_sources/example_direct_transformations.hpp
   :language: cpp
   :caption: Examples of the two direct transformation strategies enabling flexibility

Uniform velocity space binning strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A simple strategy that bins particles into uniform velocity cells is provided with VANTAGE-Reactions. It splits each of the velocity space dimensions into some number of uniform cells, up to a given extent, and in addition adds guard cells used to bin particles that might be outside of the extents. The resulting linear bin index is recorded in an integer particle property.

For example, the below strategy will bin particles into a core binning region spanning :math:`(-1.5,1.5] \times (-1.5,1.5]` split into 10 by 10 cells, and with an outer layer of guard cells capturing any particles with velocity components outside of the binning region - resulting in a total of 144 binning cells.

.. literalinclude:: ../example_sources/example_uniform_velocity_binning.hpp
   :language: cpp
   :caption: Example construction of uniform velocity binning transform

Transformation Wrappers
=======================

Often we wish to encapsulate both some marking conditions as well as the transformation we wish to perform into a single object. 
A common example is performing merging on multiple different species, but only on particles where weight is less than some threshold. In other words, the transformation we wish to perform is fixed, but only some of the marking conditions are fixed, while others vary. In this case the library offers the :class:`TransformationWrapper` class, which wraps marking conditions and a transformation strategy, and applies them to a :class:`ParticleGroup`. We can then fix the instruction "Merge all particles with weight < threshold", and can extend it with other conditions, such as "and with species ID = 1". 

.. literalinclude:: ../example_sources/example_transformation_wrapper.hpp
   :language: cpp
   :caption: An example of a :class:`TransformationWrapper` being used to remove particles with low weights for two different ID values

.. [VRANIC2015] Vranic et al. - Particle merging algorithm for PIC codes https://www.sciencedirect.com/science/article/pii/S0010465515000405

