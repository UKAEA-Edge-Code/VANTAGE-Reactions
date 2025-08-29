******************************
Reactions and their components
******************************

What reactions (the abstraction) are
====================================

As noted in the :ref:`Introduction`, we use reactions abstraction to represent the various physical collisional and reactive processes. Here we expand on those ideas and show the components of reactions as well as some examples.

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
An example of a non-linear reaction would be an elastic collision between two neutrals of the same species. There exist linearisation techniques for some of these reactions, so the initial focus of the library is linear reactions.

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

An example with the built-in charge-exchange kernels using fixed values for all of the data is given below. It demonstrates the pipeline needed to build a linear reaction object, as well as some of the method calls on the object relating to the two loops described above. More details on the individual data and kernel objects and their required properties will be presented below. 


.. literalinclude:: ../example_sources/example_linear_reaction_CX.hpp
   :language: cpp
   :caption: Constructing a CX reaction with a fixed rate and with a beam of ions

Reaction data types
===================

Reactions offers a number of built-in data types. These will be covered here in the following format:

#. Dimensionality - the number of data values produced by the data object per particle
#. Required properties - all reaction data objects provided by Reactions use the default properties enum and their required properties will be listed (both simple and species properties, where applicable)
#. Details - any explantion of the calculations done by the data object, e.g. formulae, restrictions, etc.
#. Example - where the constructor of the object is non-trivial an example of how to construct it is given

Fixed rate data
~~~~~~~~~~~~~~~

#. Dimensionality: 1
#. Required properties: none
#. Details: The rate is simply set to a fixed value
#. Example: See the example in the previous section

Fixed rate coefficient data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Dimensionality: 1
#. Required properties: Simple props: weight; Species props: none
#. Details: Given a coefficient :math:`K`, and a particle weight :math:`w`, the rate is given as :math:`Kw`, with :math:`K` being fixed.
#. Example: 

.. literalinclude:: ../example_sources/example_fixed_coeff.hpp
   :language: cpp
   :caption: Constructing a fixed rate coefficient data object

AMJUEL 1D rate fit
~~~~~~~~~~~~~~~~~~

#. Dimensionality: 1
#. Required properties: Simple props: fluid_density, fluid_temperature, weight; Species props: none 
#. Details: Uses the following fit for the rate coefficient from AMJUEL

    .. math:: 

        K=\ln\langle\sigma v\rangle = \sum_{n=0}^N b_n (\ln T)^n

where the number of coefficients :math:`N` and the coefficients :math:`b_n` are set on construction. 
The final output rate is given as :math:`nKw`, where :math:`n` here is the fluid density and :math:`w` 
is the particle weight. All normalisation is set in the constructor (see the example).
The rate is assumed to evolve some quantity :math:`q`, and requires the knowledge of the normalisation of 
that quantity. For example, if evolving the weight it should be left at 1.0 while if evolving a background 
energy field (e.g. providing an energy source) it would require the normalisation of the energy density (see below for the assumed normalisation in case of built-in kernels).
    
#. Example: 

.. literalinclude:: ../example_sources/example_amjuel1d.hpp
   :language: cpp
   :caption: Constructing an AMJUEL 1D rate fit

AMJUEL 2D rate fit (n,T)
~~~~~~~~~~~~~~~~~~~~~~~~

#. Dimensionality: 1
#. Required properties: Simple props: fluid_density, fluid_temperature, weight; Species props: none 
#. Details: Uses the following fit for the rate coefficient from AMJUEL

    .. math:: 

        K=\ln\langle\sigma v\rangle = \sum_{n=0}^N \sum_{m=0}^M \alpha_{n,m}(\ln \tilde{n})^m (\ln T)^n
        
where the numbers of coefficients :math:`N` and :math:`M`, and the coefficients :math:`\alpha_{n,m}` are set on construction. 
:math:`\tilde{n}` is density rescaled to :math:`10^{14} m^{-3}`. Density dependence is dropped below :math:`\tilde{n}=1`, and only the
:math:`m=0` coefficients are used (this is the Coronal approximation). 
The LTE limit is not implemented yet (for densities above :math:`10^{22} m^{-3}`). 
The final output rate is given as :math:`nKw`, where :math:`n` here is the fluid density and :math:`w` 
is the particle weight. Normalisation as in the 1D fit case. 

#. Example: 

.. literalinclude:: ../example_sources/example_amjuel2d.hpp
   :language: cpp
   :caption: Constructing an AMJUEL 2D rate fit

AMJUEL 2D rate fit (E,T)
~~~~~~~~~~~~~~~~~~~~~~~~

#. Dimensionality: 1
#. Required properties: Simple props: fluid_density, fluid_temperature, fluid_flow_speed, weight, velocity; Species props: none 
#. Details: Uses the following fit for the rate coefficient from AMJUEL section H.3 

    .. math:: 

        K=\ln\langle\sigma v\rangle = \sum_{n=0}^N \sum_{m=0}^M \alpha_{n,m}(\ln E)^m (\ln T)^n
        
where the numbers of coefficients :math:`N` and :math:`M`, and the coefficients :math:`\alpha_{n,m}` are set on construction. 
The neutral energy :math:`E` is relative to the fluid flow speed.
The final output rate is given as :math:`nKw`, where :math:`n` here is the fluid density and :math:`w` 
is the particle weight. Normalisation as in the 1D fit case, with the added normalisation of the velocity and the requirement for
the neutral energy to be specified in amus. 

#. Example: 

.. literalinclude:: ../example_sources/example_amjuel2dH3.hpp
   :language: cpp
   :caption: Constructing an AMJUEL 2D rate fit as a function of neutral energy

Filtered Maxwellian sampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Dimensionality: variable - corresponds to fluid flow field dimensionality 
#. Required properties: Simple props: fluid_temperature, fluid_flow_speed, velocity; Species props: none 
#. Details: Produces velocity components sampled from a drifting Maxwellian at the local fluid temperature and with local mean fluid flow. 
   Optionally filters the sampled velocities based on an interaction cross section using a rejection method, effectively sampling from

   .. math::

        f(\vec{v}) \propto \sigma(v_{rel}) f_M(\vec{v},\vec{u},T)

where :math:`\sigma(v_{rel})` is the interaction cross-section evaluated at the relative velocity :math:`|\vec{v}-\vec{u}|`, and :math:`\vec{u}`
and :math:`T` are the fluid flow speed and temperature, respectively. By default, the cross-section is assumed constant, which just leads to sampling 
from a drifting Maxwellian. See below for cross-section objects.

#. Example:

.. literalinclude:: ../example_sources/example_maxwellian_sampler.hpp
   :language: cpp
   :caption: Constructing a Maxwellian sampler in two velocity space dimensions

Cross-section objects
~~~~~~~~~~~~~~~~~~~~~

Currently, cross-section objects are restricted to being used by the above sampler. For that purpose, they can be evaluated at a given relative velocity, and 
have an associated maximum :math:`\sigma v_{rel}`, used for rejection sampling. 

Constant rate cross-section
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the simplest cross section, with :math:`\sigma v_{rel} = c`. It always accepts all samples. See the above sampler example.  

AMJUEL H.1 cross-section fit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These are single parameter fits in lab energy from AMJUEL of the form 

.. math::
    
    \ln \sigma = \sum_{n=0}^N a_n (\ln E)^n

where we convert to the centre-of-mass energy. The coefficients :math:`a_n` can be given for asymptotic values of the energy, as well, both high or low.

.. literalinclude:: ../example_sources/example_amjuel_cs.hpp
   :language: cpp
   :caption: Example of AMJUEL H.1 fit cross-section object construction

Reaction kernel types
=====================

VANTAGE-Reactions offers several built-in reaction kernels. These are presented in the following format:

#. Overview - general description, number of products, required :class:`DataCalculator` total dimensionality, etc. 
#. Required properties - the required properties from the default properties enum (as for reaction data) - here split into parent and descendant
#. Scattering kernel - if there are any products, how their velocities are calculated 
#. Weight kernel - if there are any products, how weight is distributed amongst them
#. Transformation kernel - if there are any products, how aspects of their internal states are set
#. Feedback kernel - determines how the parent weight is affected, as well as how the various source :class:`ParticleDat` values on the parent are set
#. Example - example of constructing the kernel 

Kernels that produce products have a set of specified descendant particle required properties. These are usually the particle velocities and weights, and are modified by the kernel. 
All other properties are copied from the parent, so care should be taken if some of these need zeroing (sources, etc.).

**NOTE**: Reactions assumes all sources are :class:`ParticleDat` objects on particles. All pre-built kernels also assume that the sources are not rates, i.e. that the user will divide them by the timestep 
lenght to get the rate after applying reactions. This is so that different length timesteps could be used for different reactions, or so that operator splitting can be done without worrying about the individual steps.

Base ionisation kernels
~~~~~~~~~~~~~~~~~~~~~~~

#. Overview: These are general ionisation kernels with the fewest possible assumptions. Since ionisation is an absorption process, there are no descendant particles. 
   This implementation allows for different electron, projectile, and target species, i.e. it represents projectile-impact target ionisation. It expects at least one :class:`DataCalculator` value,
   representing the energy loss rate of the projectile species in the process. Optionally, a momentum loss rate can be included, with the momentum being transferred to the target species. Electron momentum is assumed negligible. **NOTE**: The units of the energy and momentum sources are tied to the velocity normalisation via the weight and amu - e.g. the energy source normalisation is assumed to be :math:`w_0m_0 v_0^2`, where :math:`m_0` is the amu, :math:`v_0` is the velocity normalisation, and :math:`w_0` represents the weight normalisation (for example a number of particles associated with unit weight)

#. Required properties: 

   * Parent: Simple props: weight, velocity; Species props: source_density, source_energy, source_momentum
     
   * Descendant: N/A
#. Scattering kernel: N/A
#. Weight kernel: N/A
#. Transformation kernel: N/A
#. Feedback kernel: The total weight participating in the reaction is removed from the parent particle. The first :class:`DataCalculator` value is used as the energy rate, and if a momentum rate is marked as set, the second value is interpreted as that. Density sources for the target and electron species are set to that same weight value. The target momentum source always includes the momentum of the ionised neutral, regardless of the presence of a momentum kernel. 
#. Example: 

.. literalinclude:: ../example_sources/example_ionisation_kernels.hpp
   :language: cpp
   :caption: Example of constructing ionisation reaction kernels

Base charge-exchange kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Overview: These kernels perform direct charge-exchange with a pre-sampled ion. The ion velocities are assumed to be set in the accompanying :class:`DataCalculator` object. As such, this kernel is not in charge of the sampling process (use, for example, the :class:`FilteredMaxwellianSampler`). These kernels assume one reaction product, which is the resulting charge-exchanged neutral particle. **NOTE**: Energy and momentum source normalisation are the same here as in the ionisation kernels.
#. Required properties: 

   * Parent: Simple props: weight, velocity; Species props: source_density, source_energy, source_momentum
     
   * Descendant: Simple props: weight, velocity, internal_state; Species props: N/A
#. Scattering kernel: Sets the product velocities to the pre-calculated velocities from the :class:`DataCalculator`.
#. Weight kernel: The product gets the full weight that participated in the reaction
#. Transformation kernel: The products internal_state is set to the correct species ID
#. Feedback kernel: The total weight participating in the reaction is removed from the parent particle. The energy and momentum sources are computed from the participating particles' velocities (the parent and the sample ion)
#. Example: See the above example on putting together a linear reaction for an example of a CX kernel being constructed and used

Base recombination kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Overview: These kernels allow for implementing recombination using pseudo-particles (also referred to as markers). The self-consistent calculation of the rates used by the kernels is left to the users, as it depends on the mesh properties (i.e. the mapping of ion densities to the marker weights). Like the ionisation kernels, assumes that the first value calculated by the :class:`DataCalculator` is the electron energy loss rate (not including the potential energy). Similarly to the CX kernel above, this kernel assumes pre-sampled ion velocities set by a :class:`DataCalculator` (after the energy loss rate). Recombination produces a single product, and **does not** modify the weights of the parents/marker particles. **NOTE**: Energy and momentum source normalisation are the same here as in the previous two kernels.
#. Required properties:

   * Parent: Simple props: weight; Species props: source_density, source_energy, source_momentum

   * Descendant: Simple props: weight, velocitym internal_state; Species props: N/A
#. Scattering kernel: Sets the product velocities to the pre-calculated velocities from the :class:`DataCalculator` (excluding the first entry) 
#. Weight kernel:  The product gets the full weight that participated in the reaction
#. Transformation kernel:  The products internal_state is set to the correct species ID
#. Feedback kernel: Weight is **not** removed from the parent, but the particle sources is updated as if it were. The energy and momentum source of the target species (the ions) are computed from the sampled velocities. The projectile species (electron) momentum source is assumed to be negligible, while the energy cost is calculated using the energy loss rate :math:`K_E` and the normalised (to :math:`m_0 v_0^2`) ionisation potential energy :math:`\epsilon_i` as :math:`- K_E  \Delta t - \epsilon_i \Delta w`, where the timestep and weight participating in reaction are set self-consistently. 
#. Example: 

.. literalinclude:: ../example_sources/example_recombination_kernels.hpp
   :language: cpp
   :caption: Example of constructing recombination reaction kernels

Pre-built reactions
===================

VANTAGE-Reactions offers pre-built reaction classes that bundle commonly used options together. It should be noted that these can be completely reproduced by users from the base reaction class and data and kernels.

Electron-impact ionisation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Given that the most commonly treated class of ionisation reactions is electron-impact ionisation, the library offers a streamlined way of constructing electron-impact ionisation reactions. See below for an example of such a reaction. 


.. literalinclude:: ../example_sources/example_electron_impact_ion.hpp
   :language: cpp
   :caption: Example of contructing the built-in electron-impact ionisation reaction

Recombination  
~~~~~~~~~~~~~

Like electron-impact ionisation, a recombination reaction can be constructed directly without using :class:`LinearReactionBase`:

.. literalinclude:: ../example_sources/example_recombination_reaction.hpp
   :language: cpp
   :caption: Example of contructing the built-in electron-impact ionisation reaction

**NOTE**: To properly use recombination, especially with built-in AMJUEAL reaction data, care must be taken that the marker weights are updated in a way consistent with the background ion densities.

