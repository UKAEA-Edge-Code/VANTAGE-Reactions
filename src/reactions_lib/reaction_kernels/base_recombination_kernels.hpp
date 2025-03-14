#pragma once
#include <array>
#include <neso_particles.hpp>
#include "../particle_properties_map.hpp"
#include "../reaction_kernel_pre_reqs.hpp"
#include "../reaction_kernels.hpp"
#include <vector>

using namespace NESO::Particles;

namespace Reactions {

namespace BASE_RECOMB_KERNEL {
    constexpr int num_products_per_parent = 1;

    const auto props = default_properties;

    const std::vector<int> required_simple_real_props = {props.weight};

    const std::vector<int> required_species_real_props = {
        props.source_density, props.source_momentum, props.source_energy};

    const std::vector<int> required_descendant_simple_int_props = {
        props.internal_state};
    const std::vector<int> required_descendant_simple_real_props = {props.velocity,
                                                                    props.weight};
    } // namespace BASE_RECOMB_KERNEL

    template <int ndim_velocity, int ndim_source_momentum,
            bool has_momentum_req_data>
    struct RecombReactionKernelsOnDevice
        : public ReactionKernelsBaseOnDevice<
            BASE_RECOMB_KERNEL::num_products_per_parent> {
    RecombReactionKernelsOnDevice() = default;

    void scattering_kernel(
        REAL &modified_weight, Access::LoopIndex::Read &index,
        Access::DescendantProducts::Write &descendant_products,
        Access::SymVector::Write<INT> &req_int_props,
        Access::SymVector::Write<REAL> &req_real_props,
        const std::array<int, BASE_RECOMB_KERNEL::num_products_per_parent>
            &out_states,
        Access::NDLocalArray::Read<REAL, 2> &pre_req_data, double dt) const {
        for (int dimx = 0; dimx < ndim_velocity; dimx++) {
        descendant_products.at_real(index, 0, descendant_velocity_ind, dimx) =
            pre_req_data.at(index.get_loop_linear_index(), 1 + dimx);
        }
    }

    void weight_kernel(
        REAL &modified_weight, Access::LoopIndex::Read &index,
        Access::DescendantProducts::Write &descendant_products,
        Access::SymVector::Write<INT> &req_int_props,
        Access::SymVector::Write<REAL> &req_real_props,
        const std::array<int, BASE_RECOMB_KERNEL::num_products_per_parent>
            &out_states,
        Access::NDLocalArray::Read<REAL, 2> &pre_req_data, double dt) const {
        descendant_products.at_real(index, 0, descendant_weight_ind, 0) =
            modified_weight;
    }

    void transformation_kernel(
        REAL &modified_weight, Access::LoopIndex::Read &index,
        Access::DescendantProducts::Write &descendant_products,
        Access::SymVector::Write<INT> &req_int_props,
        Access::SymVector::Write<REAL> &req_real_props,
        const std::array<int, BASE_RECOMB_KERNEL::num_products_per_parent>
            &out_states,
        Access::NDLocalArray::Read<REAL, 2> &pre_req_data, double dt) const {
        descendant_products.at_int(index, 0, descendant_internal_state_ind, 0) =
            out_states[0];
    }

    void feedback_kernel(
        REAL &modified_weight, Access::LoopIndex::Read &index,
        Access::DescendantProducts::Write &descendant_products,
        Access::SymVector::Write<INT> &req_int_props,
        Access::SymVector::Write<REAL> &req_real_props,
        const std::array<int, BASE_RECOMB_KERNEL::num_products_per_parent>
            &out_states,
        Access::NDLocalArray::Read<REAL, 2> &pre_req_data, double dt) const {

        std::array<REAL, ndim_velocity> k_V_i;
        REAL visquared = 0.0;
        for (int vdim = 0; vdim < ndim_velocity; vdim++) {
        k_V_i[vdim] = pre_req_data.at(index.get_loop_linear_index(), 1 + vdim);
        visquared += k_V_i[vdim] * k_V_i[vdim];
        }

        // Set SOURCE_DENSITY
        req_real_props.at(this->projectile_source_density_ind, index, 0) -=
            modified_weight;

        req_real_props.at(this->target_source_density_ind, index, 0) -=
            modified_weight;

        // SOURCE_MOMENTUM calc
        for (int sm_dim = 0; sm_dim < ndim_source_momentum; sm_dim++) {
        req_real_props.at(this->target_source_momentum_ind, index, sm_dim) -=
            this->target_mass * modified_weight * k_V_i[sm_dim];
        }

        // set SOURCE_ENERGY
        req_real_props.at(this->target_source_energy_ind, index, 0) -=
            this->target_mass * modified_weight * visquared * 0.5;
        req_real_props.at(this->projectile_source_energy_ind, index, 0) -=
            pre_req_data.at(index.get_loop_linear_index(), 0) *
            dt - (this->normalised_potential_energy * modified_weight);
    }

    public:
    INT weight_ind;
    INT projectile_source_density_ind, projectile_source_energy_ind,
        projectile_source_momentum_ind, target_source_density_ind,
        target_source_momentum_ind, target_source_energy_ind;
    INT descendant_internal_state_ind, descendant_velocity_ind,
        descendant_weight_ind;
    REAL target_mass, normalised_potential_energy;
    };

template <int ndim_velocity = 2, int ndim_source_momentum = ndim_velocity,
        bool has_momentum_req_data = false>
struct RecombReactionKernels : public ReactionKernelsBase {
    RecombReactionKernels(
        const Species &target_species, const Species &projectile_species,
        const REAL& normalised_potential_energy, 
        std::map<int, std::string> properties_map_ = default_map)
        : ReactionKernelsBase(
                Properties<REAL>(
                    BASE_RECOMB_KERNEL::required_simple_real_props,
                    std::vector<Species>{target_species, projectile_species},
                    BASE_RECOMB_KERNEL::required_species_real_props),
                ndim_velocity + 1, properties_map_) {
        static_assert((ndim_velocity >= ndim_source_momentum),
                    "Number of dimension for VELOCITY must be greater than or "
                    "equal to number of dimensions for SOURCE_MOMENTUM.");

        auto props = BASE_RECOMB_KERNEL::props;

        this->recomb_reaction_kernels_on_device.normalised_potential_energy = normalised_potential_energy;

        this->recomb_reaction_kernels_on_device.weight_ind =
            this->required_real_props.simple_prop_index(props.weight,
                                                        this->properties_map);

        this->recomb_reaction_kernels_on_device.target_source_density_ind =
            this->required_real_props.species_prop_index(target_species.get_name(),
                                                        props.source_density,
                                                        this->properties_map);

        this->recomb_reaction_kernels_on_device.target_source_momentum_ind =
            this->required_real_props.species_prop_index(target_species.get_name(),
                                                        props.source_momentum,
                                                        this->properties_map);

        this->recomb_reaction_kernels_on_device.target_source_energy_ind =
            this->required_real_props.species_prop_index(target_species.get_name(),
                                                        props.source_energy,
                                                        this->properties_map);

        this->recomb_reaction_kernels_on_device.projectile_source_density_ind =
            this->required_real_props.species_prop_index(
                projectile_species.get_name(), props.source_density,
                this->properties_map);

        this->recomb_reaction_kernels_on_device.projectile_source_momentum_ind =
            this->required_real_props.species_prop_index(
                projectile_species.get_name(), props.source_momentum,
                this->properties_map);

        this->recomb_reaction_kernels_on_device.projectile_source_energy_ind =
            this->required_real_props.species_prop_index(
                projectile_species.get_name(), props.source_energy,
                this->properties_map);

        this->recomb_reaction_kernels_on_device.target_mass =
            target_species.get_mass();

        this->set_required_descendant_int_props(Properties<INT>(
            BASE_RECOMB_KERNEL::required_descendant_simple_int_props));

        this->set_required_descendant_real_props(Properties<REAL>(
            BASE_RECOMB_KERNEL::required_descendant_simple_real_props));

        this->recomb_reaction_kernels_on_device.descendant_internal_state_ind =
            this->required_descendant_int_props.simple_prop_index(
                props.internal_state, this->properties_map);
        this->recomb_reaction_kernels_on_device.descendant_velocity_ind =
            this->required_descendant_real_props.simple_prop_index(
                props.velocity, this->properties_map);
        this->recomb_reaction_kernels_on_device.descendant_weight_ind =
            this->required_descendant_real_props.simple_prop_index(
                props.weight, this->properties_map);

        this->set_descendant_matrix_spec<
            ndim_velocity, BASE_RECOMB_KERNEL::num_products_per_parent>();
    };

    private:
        RecombReactionKernelsOnDevice<ndim_velocity, ndim_source_momentum,
                                    has_momentum_req_data>
            recomb_reaction_kernels_on_device;

    public:
        RecombReactionKernelsOnDevice<ndim_velocity, ndim_source_momentum,
                                    has_momentum_req_data>
        get_on_device_obj() {
            return this->recomb_reaction_kernels_on_device;
        }
};
}