***************************
Particle property utilities
***************************

Properties, Species, and their usage
====================================

In order to offer tools for the standardisation of particle properties, Reactions defines several utilities. For the reaction abstraction to work, it requires knowledge of the various 
required properties of the particles. Some of these are relatively simple, and can be passed as :class:`Sym` objects to the various constructors. However, some reactions might require access to 
multiple different properties, e.g. particle velocities, weights, multiple different fields, and so on. This means that a standardised way of tracking the names and indexing of various properties 
is very helpful. Unfortunately, due to SYCL requirements (device-copyability, etc.), we are unable to use standard maps directly on device. To work around this, Reactions utilises a combination of 
a centralised (extensible) enum struct, and a mapping from the enums to the various particle property names. 

Default property enums and maps are supplied with the library, but users can extend/modify one or both. Built-in reaction data and kernels use some or all of the :class:`standard_properties_enum` entries.
These are:

#. positions
#. velocity 
#. cell_id
#. id 
#. tot_reaction_rate
#. weight
#. internal_state
#. temperature
#. density 
#. flow_speed 
#. source_energy 
#. source_momentum
#. source_density
#. fluid_density
#. fluid_temperature
#. fluid_flow_speed

If a user-defined reaction extension requires more properties than the above, the following method of extension is preferred.

.. literalinclude:: ../example_sources/example_custom_properties.hpp
   :language: cpp
   :caption: Extending the default properties enum

The above enum entries get mapped to strings used to construct :class:`Sym` objects. The default map is:

* position - "POSITION"
* velocity - "VELOCITY"
* cell_id - "CELL_ID"
* id - "ID"
* tot_reaction_rate - "TOT_REACTION_RATE"
* weight - "WEIGHT"
* internal_state - "INTERNAL_STATE"
* temperature - "TEMPERATURE"
* density - "DENSITY"
* flow_speed - "FLOW_SPEED"
* source_energy - "SOURCE_ENERGY"
* source_momentum - "SOURCE_MOMENTUM"
* source_density - "SOURCE_DENSITY"
* fluid_density - "FLUID_DENSITY"
* fluid_temperature - "FLUID_TEMPERATURE"
* fluid_flow_speed - "FLUID_FLOW_SPEED"

To create new maps with custom enums or remap the above we can use the following:

.. literalinclude:: ../example_sources/example_custom_property_map.hpp
   :language: cpp
   :caption: Extending and remaping the default property map
   
The main utility with property maps comes from the added flexibility when working with reaction abstractions. For example, a developer can write a reaction data object that uses the fluid_temperature enum,
and the user can use that object with, for example, both "ELECTRON_TEMPERATURE" and "ION_TEMPERATURE" particle properties by passing a custom map (see examples with reaction data and kernels). 

Species and the Property container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reactions offers a simple :class:`Species` class to encapsulate species data. It specifies the species name, and optionally the species mass, charge, and integer ID associated with it. Together with the :class:`Property` class, species can be used to bundle required properties for use in reaction abstractions, as well as in incremental construction of NESO-Particles :class:`ParticleSpec` objects. 

A :class:`Property` container can have simple properties (directly translated to :class:`Sym` names) and species properties, which are combined with species name to get the NESO-Particles :class:`Sym` names.
:class:`Property` containers use the property enums and maps as defined above. This allows for flexibility when defining required properties and their mapping to data stored on particles.

.. literalinclude:: ../example_sources/example_property_container.hpp
   :language: cpp
   :caption: Example of constructing Species and Property containers

ParticleSpecBuilder
~~~~~~~~~~~~~~~~~~~





