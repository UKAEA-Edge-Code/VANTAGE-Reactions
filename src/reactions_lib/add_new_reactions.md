# Instructions for adding/creating new reactions
- To add a new reaction, 3 header files(and their corresponding folders) need to be added to `src/reactions_lib`.
## Header file for ReactionData
- A header file defining the data and reaction rate calculation for a given reaction needs to be created.
- If necessary, a namespace needs to be created inside this file, storing integer vectors that define the required simple properties and the required species-dependent properties (using variable names defined in an enumerator in `src/reactions_lib/particle_properties_map.hpp`).
- There would be 2 structs inside this file, one struct is the "on-host" struct that would contain data and functions that are to be stored and used only on the host. The other struct would be the "on-device" for data and functions to be stored and used only on the SYCL device.
- Both structs need to inherit from abstract base structs. The "on-host" struct from `ReactionDataBase` and the "on-device" struct from `ReactionDataBaseOnDevice` both are defined in `src/reactions_lib/reaction_data.hpp`
- The "on-host" struct contains an instance of the "on-device" struct as a member variable and needs a corresponding getter labeled as `get_on_device_obj(...)`.
- Any parameters (or template parameters) that are needed by the "on-device" struct should be passed to the "on-host" struct and then subsequently used in the initialization of the "on-device" object in the constructor of the "on-host" struct.
- Any required properties should also be initialized in the constructor of "on-host" object. The indices of the properties (such that they can be accessed from the Sym vectors passed to `calc_rate`) should be set using the results from the appropriate `..._ prop_index(...)` methods inside the body of the "on-host" struct constructor. The indices should be integers that are public in the "on-device" struct so should be set directly.
## Header file for ReactionKernels
- A header file defining the data and kernels for a given reaction needs to be created.
- If necessary, a namespace needs to be created inside this file, storing integer vectors that define the required simple properties and the required species-dependent properties (using variable names defined in an enumerator in `src/reactions_lib/particle_properties_map.hpp`).
- There would be 2 structs inside this file, one struct is the "on-host" struct that would contain data and functions that are to be stored and used only on the host. The other struct would be the "on-device" for data and functions to be stored and used only on the SYCL device.
- Both structs need to inherit from abstract base structs. The "on-host" struct from `ReactionKernelsBase` and the "on-device" struct from `ReactionKernelsBaseOnDevice` both are defined in `src/reactions_lib/reaction_kernels.hpp`
- The "on-host" struct contains an instance of the "on-device" struct as a member variable and needs a corresponding getter labeled as `get_on_device_obj(...)`.
- Any parameters (or template parameters) that are needed by the "on-device" struct should be passed to the "on-host" struct and then subsequently used in the initialization of the "on-device" object in the constructor of the "on-host" struct.
- Any required properties should also be initialized in the constructor of "on-host" object. The indices of the properties (such that they can be accessed from the Sym vectors passed to the kernel functions) should be set using the results from the appropriate `..._ prop_index(...)` methods inside the body of the "on-host" struct constructor. The indices should be integers that are public in the "on-device" struct so should be set directly.
## Header file for Reaction
- This file should contain a Reaction object that (for now) inherits from `LinearReactionBase`. In this object's constructor there is an option to either pass all parameters (and template parameters) that `LinearReactionBase` needs but these can also be hard-coded at this level. For example, ionisation reactions hard-code the template parameter for `num_products_per_parent` to 0 which means that this template parameter doesn't need to be passed when an instance of the derived struct is initialized.

# Example for a AMJUEL 1D ionisation reaction
- 3 files are created: \
 -- `src/reactions_lib/ionisation_reactions_data/amjuel_ionisation_data.hpp`\
 -- `src/reactions_lib/ionisation_reactions_kernels/base_ionisation_kernels.hpp` (since this can be used by a fixed rate ionisation reaction as well)\
 -- `src/reactions_lib/ionisation_reactions/amjuel_ionisation.hpp`
- The reaction data file contains a namespace `AMJUEL_IONISATION_DATA` that contains the required simple properties (`fluid_density` and `fluid_temperature` in this case). Also included are 2 structs: `IoniseReactionAMJUELData` and `IoniseReactionAMJUELDataOnDevice` (note `num_coeffs` is a template parameter needed by the "on-device" struct and is passed to "on-host" which then uses it in it's initialization of an instance of the "on-device" struct).
- The reaction kernel file follows a similar pattern with a namespace `BASE_IONISATION_KERNEL` and 2 structs: `IoniseReactionKernels` and `IoniseReactionKernelsOnDevice`.
- Finally the main reaction file contains 1 struct: `IoniseReactionAMJUEL` which inherits from `LinearReactionBase` and hard-codes some parameters for the base struct (such as `num_products_per_parent` being set to 0).
- The `ReactionController.ionisation_reaction_amjuel` test shows how one would go about creating an instance of the reaction.