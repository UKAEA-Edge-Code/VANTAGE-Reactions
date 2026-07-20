.. _interpolation_usage:

****************************
Interpolation method details
****************************

.. _overview_method:

Overview of the interpolation method
====================================

To start with, a desired intermediate point (the desired function evaluation) can be thought of as having coordinates where each dimensional component is the value of a particle property (eg. locally sampled electron temperature) that would be used to calculate the function evaluation.
Initially, a bounding box (in the shape of a N-D hypercube) is constructed that encloses the intermediate point in question and whose vertices are known fixed points (function evaluations) on the rectilinear grid.

The "origin" of this bounding box is where the dimensional components of the coordinates of that point are largest that are possible whilst still being less than the dimensional components of the coordinates of the intermediate point. For example, if the intermediate point has coordinates (3.4, 2.7) then in a grid where all of the dimensions range from 0-5 with a spacing of 0.5, then the coordinates of the "origin" point for the bounding box would be (3, 2.5).

For the sake of interpretability it's useful to work with indices of the grid rather than the actual axis values. So for this example that "origin" point would have indexed coordinates of (6, 5).

Every other point in the bounding box will have coordinates that maximally differ from the "origin" point by 1 index in each direction. For example, for the (6, 5) "origin" point, the other points would have indexed coordinates (6, 6), (7, 5) and (7, 6).

Given the requirement for the axes of the grid to be sorted (monotonically increasing or decreasing), the vertices of the bounding box will have coordinates where each dimensional component is either greater or lesser than the corresponding dimensional component of the coordinates of the intermediate point.

From here, a recursive contraction is performed where linear interpolation is performed for each dimensional component. An example of a contraction from a 3D grid shown here:

.. figure:: ../figures/interpolation_visualisation.svg
   :class: with-border
   :height: 480 pt

The mathematical steps for the first contraction (for 3D->2D) would be:

 .. math::
    f(P1) = \text{linear\_interp}(x_2, V1_2, V5_2, f(V1), f(V5))\\
    f(P4) = \text{linear\_interp}(x_2, V3_2, V7_2, f(V3), f(V7))\\
    f(P2) = \text{linear\_interp}(x_2, V2_2, V6_2, f(V2), f(V6))\\
    f(P3) = \text{linear\_interp}(x_2, V4_2, V8_2, f(V4), f(V8))

The subscript for any terms going forward will denote the axis in question (x-axis - 0, y-axis - 1, z-axis - 2).
To take just the first one of these interpolations, the point :math:`P1` has no z-axis component and :math:`V1` and :math:`V5` only differ by 1 in the z-axis. Therefore the full form of the calculation of :math:`f(P1)` would be:

 .. math::
    f(P1) &= m \cdot x_2 + c \\
    m     &= \frac{f(V5) - f(V1)}{V5_2 - V1_2} \\
    c     &= f(V1) - m \cdot V1_2

This is effectively what the :math:`\text{linear\_interp(...)}` function is doing. Following this, the 2D->1D contraction would be:

 .. math::
    f(L1) = \text{linear\_interp}(x_1, P1_1, P4_1, f(P1), f(P4))\\
    f(L2) = \text{linear\_interp}(x_1, P2_1, P3_1, f(P2), f(P3))

and similarly the final contraction would be:


 .. math::
    f(x) = \text{linear\_interp}(x_0, L1, L2, f(L1), f(L2))

The subscripts for :math:`L1` and :math:`L2` are omitted since they're 1D points and the subscripts would be redundant.

Multi-dimensional function evaluations
======================================

This method can also be extended to the case where the function evaluation for each grid point might be multi-valued (as is the case for TRIM data). The flow is mostly identical but instead of evaluating for just one :math:`f(x)`, it might be evaluating for :math:`f(x)_0`, :math:`f(x)_1`, etc. (note the subscripts here are for denoting the different values of f(x) not the dimensions). Each component is interpolated individually, assuming that no component's calculation depends on another, they should only depend on the particle properties that the grid coordinates represent and potentially other external values like random numbers.

.. _cartesian_case:

Cartesian case
--------------

There are helper classes that are available in VANTAGE-Reactions for constructing the necessary grid objects for :class:`InterpolateData`. For example, if the grid in question is a standard cartesian grid of size-1 function evaluations, then :class:`GridDescriptor` and :class:`CartesianGridData` are available.

In short, the :class:`GridDescriptor` takes in an array of vectors (where each vector contains all of the coordinates of each dimension) and a lambda that is set up to return the value of the grid at a given set of coordinates. This lambda can be a computation or can just get a value from a look-up table.

The resulting :class:`GridDescriptor` object can then be passed to :class:`CartesianGridData` (which contains the crucial :func:`calc_data` in the on-device object) and this is the object that is then passed to :class:`InterpolateData`.

For convenience, the ``dims_vec`` and ``ranges_vec`` that are needed for :class:`InterpolateData` can also be retrieved from the :class:`GridDescriptor` object via getters (ie. :func:`get_interp_dims` and :func:`get_flat_ranges`).

.. _trim_case:

TRIM case
---------

There is one more helper specifically for TRIM data evaluation. The interface for the construction of :class:`TrimEvalData` from a :class:`GridDescriptor` object is similar to :class:`CartesianGridData` (but with an optional ``properties_map`` argument).

The key difference is in the construction of the :class:`GridDescriptor` object. There's an additional argument that's needed for specifying the size of each dimension associated with the TRIM tables.

The lambda used for constructing :class:`GridDescriptor` must also have a specific form, which now must return a concatenated vector that represents a TRIM table at a given set of coordinates. The length of said vector can be thought of as :math:`\sum_{i = 0}^{n}\left(\prod_{j=0}^{j=i}T_j\right)`, where :math:`n` is the number of TRIM dimensions.

For example, if the TRIM dimensions have sizes :math:`(5, 5, 5)` then the length of the vector returned by the lambda would be: :math:`5 + (5 \times 5) + (5 \times 5 \times 5) = 155`.