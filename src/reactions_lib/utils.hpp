#include <cassert>
#include <cmath>
#include <numeric>
#include <type_traits>
#include <vector>

namespace Reactions::utils{
template<typename T>
T norm2(const std::vector<T>& vec){
    static_assert(std::is_arithmetic<T>(),"Template type in norm2 must be arithmetic");
    return std::sqrt(std::accumulate(vec.begin(),vec.end(),T(),[](T a, T b){return a + b*b;}));
}

template<typename T>
std::vector<T> cross_product(const std::vector<T>& a,const std::vector<T>& b){
    static_assert(std::is_arithmetic<T>(),"Template type in cross_product must be arithmetic");
    if (a.size()!=3 || b.size()!=3){
        assert("cross_product called with vectors not of size 3");
    }

    std::vector<T> result(3);

    result[0] = a[1]*b[2] - a[2]*b[1];
    result[1] = a[2]*b[0] - a[0]*b[2];
    result[2] = a[0]*b[1] - a[1]*b[0];

    return result;
}
}