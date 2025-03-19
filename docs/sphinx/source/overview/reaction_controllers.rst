*********************
Reactions controllers
*********************

Introduction
============

While one or two reactions can be applied manually, once multiple species and multiple processes are treated, complexity is sure to explode. 
To address this, Reactions offers reaction controller objects, that bundle reactions and transformations on particles, while respecting 
certain constraints. 

For example, let us assume we wish to apply two reactions with an Euler timestep of :math:`dt=0.1`. Let us assume that the rate of the first reaction is :math:`K_1=1.0` and the rate of the second
reaction is :math:`K_2=1.0`. Now let us assume we are applying the reactions to a particle with weight :math:`w=0.01`. It is easy to see that, if reactions are applied naively and sequentially, 
all of the weight of the particle would be consumed by whichever reaction is applied first, since both :math:`K_1 dt` and :math:`K_2 dt` are greater than :math:`w`. Instead, what we would prefer to happen isthat half of the particle weight is spent on the first and half on the second reaction. This accounting is automatically performed by the reactions, and is enabled by the controllers knowing the order of operations.

Another example of potential complications is the filtering of particles based on their species/internal state, since reactions have a predefined ingoing state. This is also done automatically for each ID detected by the reaction controller amongst the reactions it is responsible for.

Finally, we might wish to perform operations on the products before they are added to the particle group, or on the reactants/parents after reactions are applied. Examples include small particle merging or removal, as well as the projection of particle properties onto the grid. 

Reaction controllers provide a centralised interface for managing the above workload. 

Deterministic reaction controller
=================================

The currently implemented reaction controller class applies all reactions to all eligible particles, splitting the reacting weight amongst the reactions proportional to their rate. The following is the sequence of actions taken by the controller, given an Euler timestep :math:`dt`:

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
