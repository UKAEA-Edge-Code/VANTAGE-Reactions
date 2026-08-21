*********************
Reactions controllers
*********************

Introduction
============

While one or two reactions can be applied manually, once multiple species and multiple processes are treated, complexity is sure to explode. 
To address this, VANTAGE-Reactions offers reaction controller objects, that bundle reactions and transformations on particles, while respecting 
certain constraints. 

For example, let us assume we wish to apply two reactions with an Euler timestep of :math:`dt=0.1`. Let us assume that the rate of the first reaction is :math:`K_1=1.0` and the rate of the second
reaction is :math:`K_2=1.0`. Now let us assume we are applying the reactions to a particle with weight :math:`w=0.01`. It is easy to see that, if reactions are applied naively and sequentially, 
all of the weight of the particle would be consumed by whichever reaction is applied first, since both :math:`K_1 dt` and :math:`K_2 dt` are greater than :math:`w`. Instead, what we would prefer to happen isthat half of the particle weight is spent on the first and half on the second reaction. This accounting is automatically performed by the reactions, and is enabled by the controllers knowing the order of operations.

Another example of potential complications is the filtering of particles based on their species/internal state, since reactions have a predefined ingoing state. This is also done automatically for each ID detected by the reaction controller amongst the reactions it is responsible for.

Finally, we might wish to perform operations on the products before they are added to the particle group, or on the reactants/parents after reactions are applied. Examples include small particle merging or removal, as well as the projection of particle properties onto the grid. 

Reaction controllers provide a centralised interface for managing the above workload. 

Reaction controller modes
=========================

Different behaviour can be obtained by passing different mode flags to the :class:`ReactionController` object. 

Currently implemented flags are:

#. standard_mode - leads to deterministic application, where all reactions are applied to all particles in accordance to their rates
#. semi_dsmc_mode - semi-deterministic Direct Simulation Monte Carlo (DSMC) mode application, where particles going through reactions are determined through random sampling, and all reactions are applied to them, completely consuming them
#. surface_mode - deterministic application, where all reactions are applied with no unreacted channel

Below are the explanations of how each mode is applied.

Deterministic reaction application
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the standard_mode flag (or no flag) is passed to the controller, it applies all reactions to all eligible particles, splitting the reacting weight amongst the reactions proportional to their rate. The following is the sequence of actions taken by the controller, given an Euler timestep :math:`dt`:

#. Filter the passed group based on particle species/internal_state 
#. Run the rate loop on all reactions, passing in the subgroups containing their respective ingoing states 
#. Run the product loop with the requested step :math:`dt`, storing all the produced particles into a separate child group
#. Perform any transformations on the child group, making sure they are species-wise
#. Perform any transformations on the post-reaction parents
#. Add the child group to the parent group, completing the application of reactions

The above is done cell-block-wise, so that buffers do not overflow when many reactions are applied. The user can control the maximum number of particles per cell (on average) and the number of cells per block. The defaults are intentionally greedy, and should be reduced in case of memory issues. 

.. literalinclude:: ../example_sources/example_reaction_controller.hpp
   :language: cpp
   :caption: Example of constructing a :class:`ReactionController` and using it with multiple reactions and transforms

Semi-DSMC reaction application
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By passing :code:`ControllerMode:semi_dsmc_mode` to the controller the following behaviour is obtained:

#. Filter the passed group based on particle species/internal_state 
#. Run the rate loop on all reactions, passing in the subgroups containing their respective ingoing states 
#. Sample uniform random numbers and compare them to :math:`1-exp(-K_{tot}dt)`, where :math:`K_{tot}` is the total reaction rate for any 
   given particle. If the sampled number is lower than that value, the particle is flagged as going through reactions 
#. Run the product loop with the requested step :math:`dt`, but only applied to the reacted particles, and using all of their weights instead of allowing
   for some unreacted fraction. This is where the method differes from the standard_mode case.
#. Perform any transformations on the child group, making sure they are species-wise
#. Perform any transformations on the post-reaction parents
#. Add the child group to the parent group, completing the application of reactions

In order to use this mode, the :code:`set_rng_kernel()` method should be called on the controller before applying it, and the RNG kernel should produce uniform numbers between 0 and 1. For an example of such a kernel see filtered Maxwellian sampling examples.

Surface mode reaction application
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By passing :code:`ControllerMode:surface_mode` to the controller the following behaviour is obtained:

#. Filter the passed group based on particle species/internal_state 
#. Run the rate loop on all reactions, passing in the subgroups containing their respective ingoing states 
#. Run the product loop using all of the particle  weights instead of allowing for some unreacted fraction. 
   This is something in between standard_mode and semi_dsmc_mode, since all particles undergo reactions.
#. Perform any transformations on the child group, making sure they are species-wise
#. Perform any transformations on the post-reaction parents
#. Add the child group to the parent group, completing the application of reactions
