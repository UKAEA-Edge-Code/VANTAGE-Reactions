// Test-aware NESOASSERT (throws under TEST_NESOASSERT=ON) must be wired
// up before the impl headers, as it was for test TUs in header mode.
#include "../../include/reactions/neso_test_assert.hpp"
#include "../../include/reactions_lib/reaction_controller_impl.hpp"
