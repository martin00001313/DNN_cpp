#include <random>
#include <vector>

#include "config.h"

class Neuron
{
public:
    Neuron(unsigned n_weights = 0) : m_weights(n_weights)
    {
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_real_distribution<> dist(0.0, 1.0);
        for (Config::float_type &i : m_weights)
        {
            i = dist(gen);
        }
        m_bias = dist(gen);
        m_state = 0.;
    }

    inline float get_weight(unsigned idx) const noexcept
    {
        if constexpr (Config::DBG_BUILD)
        {
            return (idx < m_weights.size()) ? m_weights[idx] : 0.0f;
        }
        return m_weights[idx];
    }

    inline void set_weight(unsigned idx, Config::float_type val) noexcept(Config::DBG_BUILD)
    {
        m_weights[idx] = val;
    }

    inline Config::float_type get_bias() const noexcept
    {
        return m_bias;
    }

    inline void set_bias(Config::float_type val) noexcept
    {
        m_bias = val;
    }

    inline Config::float_type get_state() const noexcept
    {
        return m_state;
    }

    inline void set_state(Config::float_type val) noexcept
    {
        m_state = val;
    }

private:
    std::vector<Config::float_type> m_weights;
    Config::float_type m_bias;
    Config::float_type m_state;
};