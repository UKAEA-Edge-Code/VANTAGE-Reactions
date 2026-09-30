#ifndef VANTAGE_REACTIONS_TYPE_ALIASES_HPP
#define VANTAGE_REACTIONS_TYPE_ALIASES_HPP

#include "test_common.hpp"

namespace VANTAGE::Reactions {

// Type aliases for the complex types used by the supported instantiations.
// These expressions are only used in unevaluated decltype contexts so they do
// not create objects or trigger template instantiations here.
using VelocityExtractor2D = decltype(extract<2>("VELOCITY"));
using WeightExtractor = decltype(extract<1>("WEIGHT"));

using SquaredWeightData =
    decltype(std::declval<FixedCoefficientData>() * extract<1>("WEIGHT"));

using VelocityReflectionPipeline2D =
    decltype(pipe(VelocityExtractor2D(NP::Sym<REAL>("VELOCITY")),
                  SpecularReflectionData<2>()));

using ScatteringDataCalculator2D =
    decltype(DataCalculator<VelocityReflectionPipeline2D>(
        std::declval<VelocityReflectionPipeline2D>()));

using VelocityExtractor3D = decltype(extract<3>("VELOCITY"));

using VelocityReflectionPipeline3D =
    decltype(pipe(VelocityExtractor3D(NP::Sym<REAL>("VELOCITY")),
                  SpecularReflectionData<3>()));

using ScatteringDataCalculator3D =
    decltype(DataCalculator<VelocityReflectionPipeline3D>(
        std::declval<VelocityReflectionPipeline3D>()));

using SphericalReflectionPipeline = decltype(pipe(
    std::declval<FixedArrayData<3>>(), SphericalBasisReflectionData()));

using CartesianReflectionPipeline = decltype(pipe(
    std::declval<FixedArrayData<3>>(), CartesianBasisReflectionData()));

using ScatteringDataCalculatorSpherical =
    decltype(DataCalculator<SphericalReflectionPipeline>(
        std::declval<SphericalReflectionPipeline>()));

using ScatteringDataCalculatorCartesian =
    decltype(DataCalculator<CartesianReflectionPipeline>(
        std::declval<CartesianReflectionPipeline>()));

using KinEnergyData2D = decltype(std::declval<WeightExtractor>() *
                                 std::declval<VelocityExtractor2D>() *
                                 std::declval<VelocityExtractor2D>());

// ---------------------------------------------------------------------------
// Linear reactions
// ---------------------------------------------------------------------------
extern template class LinearReactionBase<
    1, FixedRateData, CXReactionKernels<2>,
    DataCalculator<FixedRateData, FixedRateData>>;
extern template class LinearReactionBase<
    0, FixedRateData, IoniseReactionKernels<2>, DataCalculator<FixedRateData>>;
extern template class LinearReactionBase<
    1, FixedRateData, RecombReactionKernels<2, 2>,
    DataCalculator<FixedRateData, FixedRateData, FixedRateData>>;
extern template class LinearReactionBase<1, FixedRateData,
                                         LinearScatteringKernels<2, true>,
                                         ScatteringDataCalculator2D>;
extern template class LinearReactionBase<1, FixedRateData,
                                         LinearScatteringKernels<3, true>,
                                         ScatteringDataCalculator3D>;
// Default 4th arg (DataCalc = DataCalculator<>)
extern template class LinearReactionBase<0, FixedRateData,
                                         GeneralAbsorptionKernels<2>>;
extern template class LinearReactionBase<0, FixedRateData,
                                         SpecularReflectionKernels<2>>;
extern template class LinearReactionBase<1, FixedRateData,
                                         LinearScatteringKernels<2, false>,
                                         ScatteringDataCalculator2D>;
extern template class LinearReactionBase<
    1, FixedRateData, CXReactionKernels<2>,
    DataCalculator<FilteredMaxwellianSampler<2>>>;
extern template class LinearReactionBase<
    1, FixedRateData, CXReactionKernels<3>,
    DataCalculator<FilteredMaxwellianSampler<3>>>;
extern template class LinearReactionBase<1, FixedRateData,
                                         LinearScatteringKernels<3, true>,
                                         ScatteringDataCalculatorSpherical>;
extern template class LinearReactionBase<1, FixedRateData,
                                         LinearScatteringKernels<3, true>,
                                         ScatteringDataCalculatorCartesian>;

// ---------------------------------------------------------------------------
// Derived reactions
// ---------------------------------------------------------------------------
extern template class ElectronImpactIonisation<FixedRateData, FixedRateData, 2>;
extern template class ElectronImpactIonisation<AMJUEL1DData<9>, FixedRateData,
                                               2>;
extern template class Recombination<
    FixedRateData, DataCalculator<FixedRateData, FixedRateData, FixedRateData>,
    2>;
extern template class Recombination<
    FixedRateData,
    DataCalculator<FixedRateData, FixedRateData, FixedRateData, FixedRateData>,
    3>;

// ---------------------------------------------------------------------------
// DataCalculator specialisations
// ---------------------------------------------------------------------------
extern template class DataCalculator<FixedRateData>;
extern template class DataCalculator<FixedRateData, FixedRateData>;
extern template class DataCalculator<FixedRateData, FixedRateData,
                                     FixedRateData>;
extern template class DataCalculator<VelocityReflectionPipeline2D>;
extern template class DataCalculator<FilteredMaxwellianSampler<2>>;
extern template class DataCalculator<FilteredMaxwellianSampler<3>>;
extern template class DataCalculator<FixedRateData, FixedRateData,
                                     FixedRateData, FixedRateData>;
extern template class DataCalculator<SphericalReflectionPipeline>;
extern template class DataCalculator<CartesianReflectionPipeline>;

// ---------------------------------------------------------------------------
// Reaction data types
// ---------------------------------------------------------------------------
extern template class AMJUELFitCrossSection<2, 0, 0>;
extern template class AMJUELFitCrossSection<2, 2, 0>;
extern template class AMJUELFitCrossSection<2, 2, 2>;
extern template class AMJUELFitCrossSection<3, 3, 3>;
extern template class CellwiseReactionDataAccumulator<KinEnergyData2D>;
extern template class AMJUEL1DData<3>;
extern template class AMJUEL1DData<9>;
extern template class AMJUEL2DData<2, 2>;
extern template class AMJUEL2DDataH3<2, 2, 2>;
extern template class FixedArrayData<3>;
extern template class ArrayLookupData<1, false>;
extern template class ArrayLookupData<1, true>;
extern template class ExtractorData<1>;
extern template class ExtractorData<2>;
extern template class ExtractorData<3>;

// ---------------------------------------------------------------------------
// Interpolation / grid family (int-parametrized only)
// ---------------------------------------------------------------------------
extern template class CartesianGridData<1>;
extern template class CartesianGridData<2>;
extern template class CartesianGridData<3>;
extern template class CartesianGridData<4>;
extern template class CartesianGridData<5>;
extern template class TrimEvalData<5>;
extern template class TrimEvalData<7>;

// ---------------------------------------------------------------------------
// SWPM / pair / collision-cell subsystem
// ---------------------------------------------------------------------------
// Only the specialisations actually reached by test_pair_data_calculator.cpp,
// test_swpm_reaction.cpp and test_swpm_reaction_controller.cpp are listed.

// Pair reaction data. Whole-class instantiation is not possible for CSPairData
// specialisations whose cross section is not ConstantRateCrossSection: the
// default constructor CSPairData() delegates to CSPairData(
// ConstantRateCrossSection(0.0)), which is ill-formed for those
// CROSS_SECTION_T. The members the tests use are therefore instantiated
// individually.
extern template CSPairData<2, ConstantRateCrossSection>::CSPairData(
    ConstantRateCrossSection, std::map<int, std::string>);
extern template REAL
    CSPairData<2, ConstantRateCrossSection>::get_cs_max_rate_val(REAL);
extern template CSPairData<2, ConstantCrossSection>::CSPairData(
    ConstantCrossSection, std::map<int, std::string>);
extern template REAL
    CSPairData<2, ConstantCrossSection>::get_cs_max_rate_val(REAL);
extern template CSPairData<3, IPLCrossSection>::CSPairData(
    IPLCrossSection, std::map<int, std::string>);
extern template REAL CSPairData<3, IPLCrossSection>::get_cs_max_rate_val(REAL);

extern template class HSScatteringData<2>;
extern template class SSScatteringData<3>;
extern template class PairScatteringKernels<2>;
extern template class PairScatteringKernels<3>;

// Pair data calculators and the pair pipelines used to build them.
extern template class ConcatenatorData<CSPairData<2, ConstantRateCrossSection>,
                                       CSPairData<2, ConstantCrossSection>>;
extern template class UnaryArrayTransformData<ScalerArrayTransform<1>,
                                              PairReactionDataArgumentPack>;
extern template class PipelineData<
    CSPairData<2, ConstantRateCrossSection>,
    UnaryArrayTransformData<ScalerArrayTransform<1>,
                            PairReactionDataArgumentPack>>;
extern template class BinaryArrayTransformData<
    BinaryElementwiseOperatorTransform<1, 1, decltype(sycl::plus())>,
    CSPairData<2, ConstantRateCrossSection>,
    CSPairData<2, ConstantCrossSection>>;

extern template class PairDataCalculator<
    CSPairData<2, ConstantRateCrossSection>>;
extern template class PairDataCalculator<
    CSPairData<2, ConstantRateCrossSection>,
    CSPairData<2, ConstantCrossSection>>;
extern template class PairDataCalculator<
    ConcatenatorData<CSPairData<2, ConstantRateCrossSection>,
                     CSPairData<2, ConstantCrossSection>>>;
extern template class PairDataCalculator<
    PipelineData<CSPairData<2, ConstantRateCrossSection>,
                 UnaryArrayTransformData<ScalerArrayTransform<1>,
                                         PairReactionDataArgumentPack>>>;
extern template class PairDataCalculator<BinaryArrayTransformData<
    BinaryElementwiseOperatorTransform<1, 1, decltype(sycl::plus())>,
    CSPairData<2, ConstantRateCrossSection>,
    CSPairData<2, ConstantCrossSection>>>;
extern template class PairDataCalculator<HSScatteringData<2>>;
extern template class PairDataCalculator<SSScatteringData<3>>;

// SWPM reaction / controller / specification. Whole-class instantiation of
// SWPMReaction is not possible: it would force the default-DataCalc constructor
// SWPMReaction(..., DataCalc() = PairDataCalculator<>()), which requires
// ReactionData to be default constructible (it is not for
// HSScatteringData/SSScatteringData). The members used by the tests are
// instantiated individually instead; the DataCalc-independent members are
// declared on the SWPMReactionImpl base class and instantiated there.
extern template SWPMReaction<2, CSPairData<2, ConstantCrossSection>,
                             PairScatteringKernels<2>,
                             PairDataCalculator<HSScatteringData<2>>>::
    SWPMReaction(NP::SYCLTargetSharedPtr, std::array<int, 2>,
                 std::array<int, 2>, CSPairData<2, ConstantCrossSection>,
                 PairScatteringKernels<2>,
                 PairDataCalculator<HSScatteringData<2>>,
                 const std::map<int, std::string> &);
extern template void SWPMReactionImpl<2, CSPairData<2, ConstantCrossSection>,
                                      PairScatteringKernels<2>>::
    calculate_rates(
        NP::CellwisePairListAbsolute<NP::ParticleGroup, NP::CellwisePairList> &,
        INT, INT);
extern template void
SWPMReaction<2, CSPairData<2, ConstantCrossSection>, PairScatteringKernels<2>,
             PairDataCalculator<HSScatteringData<2>>>::
    apply(
        NP::CellwisePairListAbsolute<NP::ParticleGroup, NP::CellwisePairList> &,
        INT, INT, double, NP::ParticleGroupSharedPtr);
extern template NP::LocalArraySharedPtr<REAL> &
SWPMReactionImpl<2, CSPairData<2, ConstantCrossSection>,
                 PairScatteringKernels<2>>::get_device_rate_buffer();
extern template REAL
    SWPMReactionImpl<2, CSPairData<2, ConstantCrossSection>,
                     PairScatteringKernels<2>>::get_sigma_v_bound(REAL);
extern template std::vector<int>
SWPMReactionImpl<2, CSPairData<2, ConstantCrossSection>,
                 PairScatteringKernels<2>>::get_in_states();
extern template std::vector<int>
SWPMReactionImpl<2, CSPairData<2, ConstantCrossSection>,
                 PairScatteringKernels<2>>::get_out_states();
extern template void
SWPMReactionImpl<2, CSPairData<2, ConstantCrossSection>,
                 PairScatteringKernels<2>>::set_max_buffer_size(size_t);

extern template SWPMReaction<2, CSPairData<3, IPLCrossSection>,
                             PairScatteringKernels<3>,
                             PairDataCalculator<SSScatteringData<3>>>::
    SWPMReaction(NP::SYCLTargetSharedPtr, std::array<int, 2>,
                 std::array<int, 2>, CSPairData<3, IPLCrossSection>,
                 PairScatteringKernels<3>,
                 PairDataCalculator<SSScatteringData<3>>,
                 const std::map<int, std::string> &);
extern template void
SWPMReactionImpl<2, CSPairData<3, IPLCrossSection>, PairScatteringKernels<3>>::
    calculate_rates(
        NP::CellwisePairListAbsolute<NP::ParticleGroup, NP::CellwisePairList> &,
        INT, INT);
extern template void
SWPMReaction<2, CSPairData<3, IPLCrossSection>, PairScatteringKernels<3>,
             PairDataCalculator<SSScatteringData<3>>>::
    apply(
        NP::CellwisePairListAbsolute<NP::ParticleGroup, NP::CellwisePairList> &,
        INT, INT, double, NP::ParticleGroupSharedPtr);
extern template NP::LocalArraySharedPtr<REAL> &
SWPMReactionImpl<2, CSPairData<3, IPLCrossSection>,
                 PairScatteringKernels<3>>::get_device_rate_buffer();
extern template REAL
    SWPMReactionImpl<2, CSPairData<3, IPLCrossSection>,
                     PairScatteringKernels<3>>::get_sigma_v_bound(REAL);
extern template std::vector<int>
SWPMReactionImpl<2, CSPairData<3, IPLCrossSection>,
                 PairScatteringKernels<3>>::get_in_states();
extern template std::vector<int>
SWPMReactionImpl<2, CSPairData<3, IPLCrossSection>,
                 PairScatteringKernels<3>>::get_out_states();
extern template void
SWPMReactionImpl<2, CSPairData<3, IPLCrossSection>,
                 PairScatteringKernels<3>>::set_max_buffer_size(size_t);

extern template class SWPMReactionController<
    SWPMDSMCSpecification, NP::HostRNGGenerationFunction<REAL>>;
extern template class AbstractSWPMSpecification<
    NP::HostAtomicBlockKernelRNG<REAL>>;

} // namespace VANTAGE::Reactions
#endif