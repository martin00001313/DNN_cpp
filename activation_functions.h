#include <cmath>

#include "config.h"

namespace activation_funtions
{
using float_type = Config::float_type;

constexpr float_type sigmoid(float_type activation)
{
    return static_cast<float_type>(1.f) / (static_cast<float_type>(1. + std::exp(-activation)));
}

constexpr float_type derivative_sigmoid(float_type val)
{
    return val * (static_cast<float_type>(1.) - val);
}

} // namespace activation_funtions