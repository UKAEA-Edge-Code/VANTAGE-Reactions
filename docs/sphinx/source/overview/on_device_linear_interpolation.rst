********************************************
On-device N-dimensional Linear Interpolation
********************************************

Introduction
============

When there are cases where analytical function evaluations from particle properties are not tenable (eg. TRIM data), pre-calculated grids of function evaluations at a limited set of discrete points are usually the method used to approach the problem. Each dimensional component of the coordinates of these points corresponds to a value of a particle property, such as the locally sampled fluid density or locally sampled fluid temperature ("locally sampled" here just means sampled at the particle location). A pre-requisite for the pre-calculated grid is that each axis must be sorted to be either monotonically increasing or decreasing.

The method available in VANTAGE-Reactions to retrieve arbitrary function evaluations from a pre-calculated grid is to use a series of nested hypercube contractions via 1D linear interpolations. The general method is outlined in this paper Murman_S_apnum_jun13_.

.. _Murman_S_apnum_jun13: https://www.nas.nasa.gov/assets/nas/pdf/staff/Murman_S_apnum_jun13.pdf

A pre-requisite for the axes of the grid is that they must be monotonic for the hypercube construction to work correctly.

This method can be extended to any arbitrary number of dimensions and scales (in terms of linear interpolation evaluations) as :math:`O(2^n - 1)` where :math:`n` is the number of dimensions. For further details on the underlying maths: :ref:`overview_method`

Limitations
===========

Whilst being quite powerful, this method does have some limitations. Chiefly, it's quite sensitive to the sparseness of the pre-calculated grid. If the grid is too sparse then the linear interpolation will introduce inaccuracies that effectively come from the approximation of a constant linear gradient between points in the hypercube.

Grid construction
=================

There is an expectation that the :class:`ReactionDataBase`-derived object that is an input of :class:`InterpolateData` has the correct construction in the sense that the on-device :func:`calc_data` has the correct form of arguments and outputs. Specifically the form of :func:`calc_data` should follow the second definition in :class:`ReactionDataBaseOnDevice`.

The :func:`calc_data` should provide a function evaluation at a given point, :class:`InterpolateData` will use values from multiple invocations of :func:`calc_data` to construct the hypercube which will have function evaluations at each vertex.

Inputs for :class:`InterpolateData`
-----------------------------------

For flexibility, whilst there are guide-rails for what's expected from the input :class:`ReactionDataBase`, full manual configuration and usage is possible. 

But for 2 common cases, there are some helper classes available, that aim to reduce the friction in constructing the input object.

Cartesian case
--------------

In the case that the pre-calculated grid is a simple cartesian grid of size-1 function evaluations, :class:`CartesianGridData` is available. An object of this type can be pre-constructed and then simply passed as an argument for the construction of a :class:`InterpolateData` object.

If you already have the pre-requisite flattened vectors then it's possible to directly construct :class:`CartesianGridData`. But there's an additional helper class that can aid in producing the flattened vectors. :class:`GridDescriptor` only requires an array of vectors (individual vectors that contain coordinates for each axis) and a lambda that returns function evaluations given a set of coordinates. The constructed :class:`GridDescriptor` object can then be used to construct :class:`CartesianGridData`.

For further details: :ref:`cartesian_case`

TRIM case
--------------------------

Similarly, if the pre-calculated grid is a cartesian grid of multi-dimensional function evaluations (specifically in the format of TRIM tables), then :class:`TrimEvalData` is available and functions much the same way as :class:`CartesianGridData`.

If the correct template arguments and standard arguments are given to :class:`GridDescriptor` then it can also provide the necessary flattened vectors for :class:`TrimEvalData`.

There are some further details (including the expected form of the lambda provided to :class:`GridDescriptor`) in: :ref:`trim_case`

Usage (size-1 function)
=======================

The implementation of the interpolation is in :class:`InterpolateData` which inherits from :class:`CompositeData`. This is due to how the access to the pre-calculated grid is managed. For flexibility, rather than passing a full grid to the host-side :class:`InterpolateData` object, a :class:`ReactionDataBase`-derived object is passed, which will have a :func:`calc_data` that can retrieve values at coordinates that correspond to particle property values as described in `Grid construction`_. This can be a simple retrieval from a look-up table or something more complicated like doing a limited set of calculations or even incorporating samples from a random distribution.

A few pre-requisites for the construction of :class:`InterpolateData`:

  - ``output_ndim``, a ``size_t`` template parameter that specifies the number of outputs of the grid function. In the case of a single-valued function this would be 1.
  - ``interp_ndim``, a ``size_t`` template parameter that specifies how many dimensions out of the total number of dimensions of the grid to interpolate.
  - ``non_interp_ndim``, a ``size_t`` template parameter that is effectively the inverse of ``interp_ndim`` (default value is 0).
  - ``dims_vec`` , a ``std::vector<size_t>`` that contains the lengths of the extent of each dimension in the grid.
  - ``coords_vec``, a ``std::vector<REAL>`` that is a vector containing a concatenated list of all of values of each dimension in the same order as ``dims_vec``. For example if ``dims_vec = {3, 2}`` then ``coords_vec = {dim0_val_0, dim0_val_1, dim0_val_2, dim1_val_0, dim1_val_1}``.
  - ``interp_indices``, a ``std::array<REAL, interp_ndim>`` that just specifies which of the dimensions of the grid are to be interpolated.
  - ``extrapolation_type``, an enum specifying the choice of how to handle extrapolation as explained in `Extrapolation`_.

There are a few narrow interfaces for :class:`InterpolateData` but in the example below, the widest interface is shown:

.. literalinclude:: ../example_sources/example_interpolation.hpp
  :language: cpp
  :caption: Example of constructing an InterpolateData object.

Usage (multi-dimensional function)
==================================

For multi-valued functions the process is similar but has a few key differences. Firstly, the helper functions are handled differently but also the constructor for :class:`InterpolateData` requires an extra argument that specifies which dimensions of the grid that need to be interpolated. This is due to the fact that the "grid" for multi-valued functions is treated as having (grid_dimensions + n_function_outputs). For example, with TRIM data, if the tables were assigned to coordinates with dimensionality of 2 but the access convention for the table required 3 numbers (and also outputted 3 numbers) then the grid's dimensionality is 5.

An example is shown here:

.. literalinclude:: ../example_sources/example_trim_interpolation.hpp
  :language: cpp
  :caption: Example of constructing an InterpolateData object (specifically for TRIM data).

Extrapolation
=============

If the desired point has coordinates either partially or fully outside of the valid range of any/all dimensions of the pre-calculated grid then a choice must be made as to how to handle this. The options for this in VANTAGE-Reactions are:

  - **default**: Continue with the linear interpolation using the gradient and intercept calculation from the edge of the grid.
  - `ExtrapolationType::clamp_to_zero`: Clamp the function evaluation to zero if any dimensional components of the coordinates of the desired point are outside the grid.
  - `ExtrapolationType::clamp_to_edge`: Clamp the function evaluation to be as if it were calculated from the grid points at the edge of the grid (no gradient or intercept continuation).
  
Integration with Reactions
==========================

The wrapped interpolation pipeline is effectively a composite :class:`ReactionData` object whose generated values can be treated as either inputs for another :class:`ReactionData` or as direct pre-requisite data for :class:`ReactionKernels`. The TRIM data example, would pass the values directly to a :class:`ReactionKernels` to decide velocities via :func:`scattering_kernel`. For more details on construction of compatible interpolation pipelines see: :ref:`interpolation_usage`.
