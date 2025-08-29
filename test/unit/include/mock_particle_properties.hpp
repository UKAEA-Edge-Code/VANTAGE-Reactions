#ifndef REACTIONS_MOCK_PARTICLE_PROPERTIES_H
#define REACTIONS_MOCK_PARTICLE_PROPERTIES_H
#include <neso_particles.hpp>
#include <reactions/reactions.hpp>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

namespace PropertiesTest {
inline auto int_props = Properties<INT>(
    std::vector<int>{default_properties.id, default_properties.internal_state,
                     default_properties.cell_id});

inline auto real_props = Properties<REAL>(
    std::vector<int>{
        default_properties.position, default_properties.velocity,
        default_properties.tot_reaction_rate, default_properties.weight,
        default_properties.fluid_density, default_properties.fluid_temperature,
        default_properties.fluid_flow_speed},
    std::vector<Species>{Species("ELECTRON")},
    std::vector<int>{
        default_properties.temperature, default_properties.density,
        default_properties.flow_speed, default_properties.source_energy,
        default_properties.source_density, default_properties.source_momentum});

struct CustomPropertiesEnum : StandardPropertiesEnum {
public:
  enum {
    test_custom_prop1 = default_properties.fluid_flow_speed + 1,
    test_custom_prop2
  };
};

inline auto custom_props = CustomPropertiesEnum();

struct custom_prop_map_struct {
  custom_prop_map_struct() {
    this->private_map[custom_props.test_custom_prop1] = "TEST_PROP1";
    this->private_map[custom_props.test_custom_prop2] = "TEST_PROP2";
  }

  std::map<int, std::string> get_map() { return this->private_map; }

private:
  std::map<int, std::string> private_map = get_default_map();
};

inline auto custom_prop_map =
    PropertiesMap(custom_prop_map_struct().get_map()).get_map();

struct custom_prop_map_no_weight_struct {
  custom_prop_map_no_weight_struct() {
    this->private_map[custom_props.test_custom_prop1] = "TEST_PROP1";
    this->private_map[custom_props.test_custom_prop2] = "TEST_PROP2";
    this->private_map.extract(default_properties.weight);
  }

  std::map<int, std::string> get_map() { return this->private_map; }

private:
  std::map<int, std::string> private_map = get_default_map();
};

inline auto custom_prop_no_weight_map =
    PropertiesMap(custom_prop_map_no_weight_struct().get_map()).get_map();

} // namespace PropertiesTest
#endif