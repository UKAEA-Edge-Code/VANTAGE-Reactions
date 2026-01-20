******************************
Reactions and their components
******************************

The reaction abstraction
========================

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

Reaction kernels and the product loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With any rate data required to apply the reaction calculated and stored for some particles, the next step is to apply the reaction, which might involve feedback on fields and the ingoing particle, as well as some specification of product properties following the reaction. This is (semi-)independently specified by choosing a reaction kernel. The only requirement on the data that a kernel might have is that any required data exists, i.e. that the total dimensionality of data conforms with whatever the kernel requires. For example, if a kernel requires 2 sampled velocities, the :class:`DataCalculator` must produce a total of 2 data values per perticle. Other than this requirement, data and kernels are independent. Note that data calculated by the :class:`DataCalculator` is calculated at the time of application of the reaction. This is to avoid unnecessary computations (for example when randomly selecting which particles have reacted we do not want to calculate all of the data for particles that did not react).

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

As noted above, reaction data objects are used both for reaction rate calculations as well as any other data that might be required for the desired application of reactions.

Broadly, the data can be split up into the following groups:

#. Rate-specific data - data originally intended to be used for various reaction rate calculations, including other size 1 data
#. Multi-dimensional data - data used for such things as sampling velocities from a distribution
#. Composite data - data objects that represent compositions of other data objects. These allow for such things as pipeline construction
#. Surface reaction data - objects designed for calculating various surface interaction data, such as post-reflection velocities

Reactions offers a number of built-in data types. These will be covered here in the following format:

#. Dimensionality - the number of data values produced by the data object per particle (if the data is a composite and requires a specific dimensionality of input data this will be noted here)
#. Required properties - all reaction data objects provided by Reactions use the default properties enum and their required properties will be listed (both simple and species properties, where applicable)
#. Details - any explantion of the calculations done by the data object, e.g. formulae, restrictions, etc.
#. Example - where the constructor of the object is non-trivial an example of how to construct it is given

Rate-specific and size-1 data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following data objects all return size 1 data, and the majority of them are meant to be used as reaction rate data. 

Fixed rate data
^^^^^^^^^^^^^^^

#. Dimensionality: 1
#. Required properties: none
#. Details: The rate is simply set to a fixed value :math:`K`, so that the weight evolution equation (assuming deterministic evolution) is:

.. math::

   \frac{dw}{dt} = -K


#. Example: See the example in the previous section

Fixed rate coefficient data
^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Dimensionality: 1
#. Required properties: Simple props: weight; Species props: none
#. Details: Given a coefficient :math:`k`, and a particle weight :math:`w`, the rate is given as :math:`kw`, with :math:`k` being fixed. Leads to the following deterministic weight evolution equation

.. math::

   \frac{dw}{dt} = -kw

#. Example: 

.. literalinclude:: ../example_sources/example_fixed_coeff.hpp
   :language: cpp
   :caption: Constructing a fixed rate coefficient data object

AMJUEL 1D rate fit
^^^^^^^^^^^^^^^^^^

#. Dimensionality: 1
#. Required properties: Simple props: fluid_density, fluid_temperature, weight; Species props: none 
#. Details: Uses the following fit for the rate coefficient from AMJUEL

    .. math:: 

        k=\ln\langle\sigma v\rangle = \sum_{n=0}^N b_n (\ln T)^n

where the number of coefficients :math:`N` and the coefficients :math:`b_n` are set on construction. 
The final output rate is given as :math:`nKw`, where :math:`n` here is the fluid density and :math:`w` 
is the particle weight. All normalisation is set in the constructor (see the example).
The rate is assumed to evolve some quantity :math:`q`, and requires the knowledge of the normalisation of 
that quantity. For example, if evolving the weight it should be left at 1.0 while if evolving a background 
energy field (e.g. providing an energy source) it would require the normalisation of the energy density (see below for the assumed normalisation in case of built-in kernels). When used as the deterministic reaction rate (evolving weight), leads to


    .. math::

        \frac{dw}{dt} = -nkw
    
#. Example: 

.. literalinclude:: ../example_sources/example_amjuel1d.hpp
   :language: cpp
   :caption: Constructing an AMJUEL 1D rate fit

AMJUEL 2D rate fit (n,T)
^^^^^^^^^^^^^^^^^^^^^^^^

#. Dimensionality: 1
#. Required properties: Simple props: fluid_density, fluid_temperature, weight; Species props: none 
#. Details: Uses the following fit for the rate coefficient from AMJUEL

    .. math:: 

        k=\ln\langle\sigma v\rangle = \sum_{n=0}^N \sum_{m=0}^M \alpha_{n,m}(\ln \tilde{n})^m (\ln T)^n
        
where the numbers of coefficients :math:`N` and :math:`M`, and the coefficients :math:`\alpha_{n,m}` are set on construction. 
:math:`\tilde{n}` is density rescaled to :math:`10^{14} m^{-3}`. Density dependence is dropped below :math:`\tilde{n}=1`, and only the
:math:`m=0` coefficients are used (this is the Coronal approximation). 
The LTE limit is not implemented yet (for densities above :math:`10^{22} m^{-3}`). 
The final output rate is given as :math:`nKw`, where :math:`n` here is the fluid density and :math:`w` 
is the particle weight. Normalisation and effective deterministic evolution equation as in the 1D fit case. 

#. Example: 

.. literalinclude:: ../example_sources/example_amjuel2d.hpp
   :language: cpp
   :caption: Constructing an AMJUEL 2D rate fit

AMJUEL 2D rate fit (E,T)
^^^^^^^^^^^^^^^^^^^^^^^^

#. Dimensionality: 1
#. Required properties: Simple props: fluid_density, fluid_temperature, fluid_flow_speed, weight, velocity; Species props: none 
#. Details: Uses the following fit for the rate coefficient from AMJUEL section H.3 

    .. math:: 

        K=\ln\langle\sigma v\rangle = \sum_{n=0}^N \sum_{m=0}^M \alpha_{n,m}(\ln E)^m (\ln T)^n
        
where the numbers of coefficients :math:`N` and :math:`M`, and the coefficients :math:`\alpha_{n,m}` are set on construction. 
The neutral energy :math:`E` is relative to the fluid flow speed.
The final output rate is given as :math:`nKw`, where :math:`n` here is the fluid density and :math:`w` 
is the particle weight. Normalisation as in the 1D fit case, with the added normalisation of the velocity and the requirement for
the neutral energy to be specified in amus. Deterministic evolution equation as in the 1D fit case.

#. Example: 

.. literalinclude:: ../example_sources/example_amjuel2dH3.hpp
   :language: cpp
   :caption: Constructing an AMJUEL 2D rate fit as a function of neutral energy

Arrhenius data
^^^^^^^^^^^^^^

#. Dimensionality: 1
#. Required properties: Simple props: weight, fluid_temperature; Species props: none
#. Details: Given two coefficients :math:`a` and :math:`b`, returns an Arrhenius form rate :math:`a T^b w`, with :math:`T` being a temperature, and :math:`w` being the particle weight. **NOTE**: for many reactions this will need to be multiplied by one or more densities - see below entries on composite data for how one can do that. Leads to the following deterministic weight evolution equation (assuming no additional density multiplication)

.. math::

   \frac{dw}{dt} = -aT^bw

#. Example: 

.. literalinclude:: ../example_sources/example_arrhenius.hpp
   :language: cpp
   :caption: Constructing an Arrhenius rate data

Sampler data
^^^^^^^^^^^^

#. Dimensionality: 1
#. Required properties: Simple props: none; Species props: none
#. Details: Returns a sample from the contained random number generator kernel. Not meant to be used as a rate data object. Instead, it was implemented as a way of allowing random sample input to composite data objects.
#. Example: 

.. literalinclude:: ../example_sources/example_sampler.hpp
   :language: cpp
   :caption: Constructing a sampler data object

Multi-dimensional data
~~~~~~~~~~~~~~~~~~~~~~

The following data objects allow for generating multidimensional data, with the most common use case being velocity generation (e.g. post-scattering values), or calculating inputs into composite data objects.

Fixed array data
^^^^^^^^^^^^^^^^

#. Dimensionality: any
#. Required properties: Simple props: none; Species props: none
#. Details: Returns a fixed array of values. Useful for providing fixed inputs to composite data objects.
#. Example: 

.. literalinclude:: ../example_sources/example_fixed_array_data.hpp
   :language: cpp
   :caption: Constructing a simple fixed array data object

Array lookup data
^^^^^^^^^^^^^^^^^

#. Dimensionality: any
#. Required properties: Simple props: custom key (specified by :class:`Sym`); Species props: none
#. Details: Uses an integer property on the particle as a key for a map, returning arrays based on the value of the property, with the option to specify a default return value if the key isn't in the map. 
#. Example: 

.. literalinclude:: ../example_sources/example_array_lookup.hpp
   :language: cpp
   :caption: Constructing a simple fixed array data object

Extractor data
^^^^^^^^^^^^^^

#. Dimensionality: variable (up to the dimensionality of the extracted property)
#. Required properties: Simple props: custom key (specified by :class:`Sym`); Species props: none
#. Details: Returns the first N components of a given REAL particle property. Useful for providing inputs into composite data objects.
#. Example: 

.. literalinclude:: ../example_sources/example_extractor.hpp
   :language: cpp
   :caption: Constructing an extractor data object

Filtered Maxwellian sampler
^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^

Currently, cross-section objects are restricted to being used by the above sampler. For that purpose, they can be evaluated at a given relative velocity, and 
have an associated maximum :math:`\sigma v_{rel}`, used for rejection sampling. 

Constant rate cross-section
---------------------------

This is the simplest cross section, with :math:`\sigma v_{rel} = c`. It always accepts all samples. See the above sampler example.  

AMJUEL H.1 cross-section fit
----------------------------

These are single parameter fits in lab energy from AMJUEL of the form 

.. math::
    
    \ln \sigma = \sum_{n=0}^N a_n (\ln E)^n

where we convert to the centre-of-mass energy. The coefficients :math:`a_n` can be given for asymptotic values of the energy, as well, both high or low.

.. literalinclude:: ../example_sources/example_amjuel_cs.hpp
   :language: cpp
   :caption: Example of AMJUEL H.1 fit cross-section object construction

Composite data
~~~~~~~~~~~~~~

The following data objects all have in common that they contain other data objects, and that they perform all operations on device, i.e. without saving intermediate states in a buffer such as the one used by the :class:`DataCalculator`.

Since the purely composite data objects in this section require particle properties only through their contained objects, that section of the description will be omitted for brevity. 

Concatenator data
^^^^^^^^^^^^^^^^^

#. Dimensionality: sum of the dimensions of the contained objects
#. Details: Takes as aruments any number of data objects, evaluates their outputs, and concatenates the result, similarly to the effects of the :class:`DataCalculator`, except fully on device.
#. Example: 

.. literalinclude:: ../example_sources/example_concatenator.hpp
   :language: cpp
   :caption: Example of concatenating two simple data objects

Pipeline data
^^^^^^^^^^^^^

#. Dimensionality: the output dimensionality of the final step in the pipeline
#. Details: Takes as arguments any number of data objects, and passes the outputs from left to right. The data objects must have compatible input/output dimensions (see example). All calculations are performed on device.
#. Example: 

.. literalinclude:: ../example_sources/example_pipeline.hpp
   :language: cpp
   :caption: Example of piping a result from one data object into another

Array transform data
^^^^^^^^^^^^^^^^^^^^

The following data objects perform unary or binary transformation on arrays, allowing for composition of simple operations.

Unary array transform data
--------------------------


#. Dimensionality: varies based on the transformation applied - input dimension generally non-zero, so these objects must be part of a pipeline
#. Details: Allow taking the result of another data object and applying a unary transformation on it, returning the result. A number of transformations are implemented - see example below
#. Example: 


.. literalinclude:: ../example_sources/example_unary_array_transform_data.hpp
   :language: cpp
   :caption: Various examples of unary array transform data objects available

Binary array transform data
---------------------------

#. Dimensionality: varies based on the transformation applied 
#. Details: Allows taking two reaction data objects and applying a binary transformation on their result. A number of transformations are implemented - see example below
#. Example: 

.. literalinclude:: ../example_sources/example_binary_array_transform_data.hpp
   :language: cpp
   :caption: Various examples of binary array transform data objects available

**EXPERIMENTAL** Lambda wrappers
----------------------------------

The above unary and binary transform data all rely on built-in transforms or standard operators. There are situations in which those do not allow enough flexibility, so VANTAGE-Reactions offers a wrapper for lambda functions that can be used in binary and unary array transform data objects.

.. WARNING::
   This is an **experimental** feature designed to provide device-copyable lambdas.
   While it works on some common backends, due to the nature of the
   workaround, there is **at least one known issue** with the generic adaptivecpp backend! 
   Future work is planned on adressing this, but the wrappers shouldn't currently be
   used in production!


.. literalinclude:: ../example_sources/example_lambda_wrapper_array_transform_data.hpp
   :language: cpp
   :caption: Various examples of the experimental lambda-based array transform data

Surface reaction data
~~~~~~~~~~~~~~~~~~~~~

The data objects in this section are specialised for surface reaction uses, meaning that they require surface interaction data to be present on the particles.

Specular reflection data
^^^^^^^^^^^^^^^^^^^^^^^^

#. Dimensionality: variable - depends on the velocity dimension (2 or 3). Requires input of the same dimension, representing the ingoing velocities.
#. Required properties: Simple props: boundary intersection normal; Species props: none
#. Details: Given a surface normal and an input velocity, reflects the velocity specularly based ont the normal. Should be used as part of a pipeline, allowing for modification of input and output velocities.
#. Example - see pipeline example 

Spherical basis reflection data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Dimensionality: 3. Requires input of the same dimension, representing the output velocity in (:math:`v`, :math:`\theta`, :math:`\varphi`), where the first entry is the velocity magnitude, the second the angle with respect to the surface normal, and the third the angle with respect to the initial (pre-reflection) velocity projection onto the surface
#. Required properties: Simple props: velocity, boundary intersection normal; Species props: none
#. Details: Given the reflected velocity in spherical coordinates, uses the particle velocity and the surface normal to construct a local basis for reflection. Useful when reflected data is given in spherical coordinates (such as from the TRIM database)
#. Example: 


.. literalinclude:: ../example_sources/example_spherical_basis_reflection.hpp
   :language: cpp
   :caption: Constructing a SphericalBasisReflectionData object with fixed post-collision velocities

Cartesian basis reflection data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. Dimensionality: 3. Requires input of the same dimension, representing the output velocity in the Cartesian basis defined by the ingoing velocity and the surface normal. The first two components are parallel to the surface (with the first component in the direction determined by the projection of the ingoing particle velocity). The final component is in the direction of the surface normal (directed back into the domain). 
#. Required properties: Simple props: velocity, boundary intersection normal; Species props: none
#. Details: Given the velocity and surface normal, determines the local basis and sets the outgoing particle velocity based on the input Cartesian components. Useful when the reflected data is given in Cartesian coordinates (such as for thermal reflection)
#. Example: 

.. literalinclude:: ../example_sources/example_cartesian_basis_reflection.hpp
   :language: cpp
   :caption: Constructing a CartesianBasisReflectionData object with fixed post-collision velocities

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

General absorption kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Overview: These kernels represent a general absorption process, i.e. anything that removes the weight of particles without creating new particles. Unlike ionisation, it only stores the particle, momentum, and energy sources due to the absorbed particle, and not due to any of the particles that might be interacting with it.
#. Required properties:

   * Parent: Simple props: weight, velocity, source_density, source_energy, source_momentum; Species props: N/A

   * Descendant: N/A; Species props: N/A
#. Scattering kernel: N/A
#. Weight kernel:  N/A
#. Transformation kernel:  N/A
#. Feedback kernel: The weight is removed from the parent, and together with the velocity of the particle it determines all three sources. **NOTE**: Unlike specific kernels, the sources are not species-specific, which means that property remapping is required. This is particularly important when using these kernels to specify surface processes.
#. Example: 

.. literalinclude:: ../example_sources/example_general_absorption_kernels.hpp
   :language: cpp
   :caption: Example of constructing general absorption kernels

General linear scattering kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Overview: These kernels produce one product, and the velocities of the product are set by the `DataCalculator` values. Sources (momentum and energy) are then calculated based on the parent and product velocities and the reacted weight.
#. Required properties:

   * Parent: Simple props: weight, velocity, source_energy, source_momentum; Species props: N/A

   * Descendant: internal_state, velocity, weight; Species props: N/A
#. Scattering kernel: The velocities of the product are set from the values calculated by the `DataCalculator` of the containing reaction
#. Weight kernel:  All reacted weight is passed onto the product
#. Transformation kernel:  The product internal_state is set from the outgoing particle ID
#. Feedback kernel: The weight is removed from the parent, and the momentum and energy sources are calculated using the parent and product velocities. **NOTE**: Unlike specific kernels, the sources are not species-specific, which means that property remapping is required. This is particularly important when using these kernels to specify surface processes.
#. Example: 

.. literalinclude:: ../example_sources/example_general_linear_scattering_kernels.hpp
   :language: cpp
   :caption: Example of constructing general linear scattering kernels and using them to create a specular reflection reaction

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

