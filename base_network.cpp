#include "base_network.h"

DNN_base::DNN_base(const std::vector<unsigned> &size_of_layers)
{
    if constexpr (Config::DBG_BUILD)
    {
        assert(!size_of_layers.empty());
        for (const auto &i : size_of_layers)
        {
            assert(i > 0);
        }
    }

    m_weigths_of_layers.resize(size_of_layers.size());
    m_weigths_of_layers.front() = std::vector<Neuron>(size_of_layers[0]);

    std::vector<Neuron> layer;
    for (size_t i = 1; i < size_of_layers.size(); ++i)
    {
        const size_t neurons_count = size_of_layers[i];
        const size_t weights_count = size_of_layers[i - 1];
        layer.reserve(size_of_layers[i]);
        for (size_t j = 0; j < neurons_count; ++j)
        {
            layer.emplace_back(weights_count);
        }
    }
}

void DNN_base::initialize_input_layer(const base_container states) noexcept(Config::DBG_BUILD)
{
    std::vector<Neuron> &input_layer = *(m_weigths_of_layers.begin());
    if constexpr (Config::DBG_BUILD)
    {
        assert(states.size() == input_layer.size());
    }

    for (unsigned idx = 0; idx < states.size(); ++idx)
    {
        input_layer[idx].set_state(states[idx]);
    }
}

Config::float_type DNN_base::calculate_activation(unsigned layer_idx, unsigned neuron_idx) noexcept(Config::DBG_BUILD)
{
    if constexpr (Config::DBG_BUILD)
    {
        assert(layer_idx > 0 && layer_idx < m_weigths_of_layers.size());
    }

    const auto &prev_layer = m_weigths_of_layers[layer_idx - 1];
    const Neuron &neuron = m_weigths_of_layers[layer_idx][neuron_idx];
    Config::float_type res = neuron.get_bias();
    for (size_t i = 0; i < prev_layer.size(); ++i)
    {
        res += prev_layer[i].get_state() * neuron.get_weight(i);
    }
    return activation_funtions::sigmoid(res);
}

void DNN_base::forward_propagation() noexcept(Config::DBG_BUILD)
{
    for (unsigned i = 1; i < m_weigths_of_layers.size(); ++i)
    {
        std::vector<std::future<void>> per_layer_futures;
        const auto &cur_layer = m_weigths_of_layers[i];
        per_layer_futures.reserve(cur_layer.size());
        for (size_t j = 0; j < cur_layer.size(); ++j)
        {
            per_layer_futures.emplace_back(std::async([&net = *this, layer_idx = i, neuron_idx = j]() {
                net.m_weigths_of_layers[layer_idx][neuron_idx].set_state(net.calculate_activation(layer_idx, neuron_idx));
            }));
        }
        for (auto &ft : per_layer_futures)
        {
            ft.get();
        }
    }
}

auto DNN_base::calculate_delta_of_outputs(const base_container &expected) const noexcept(Config::DBG_BUILD) -> base_container
{
    base_container delta(expected.size());
    for (size_t i = 0; i < expected.size(); ++i)
    {
        const auto output = m_weigths_of_layers.back()[i].get_state();
        delta[i] = (expected[i] - output) * activation_funtions::derivative_sigmoid(output);
    }
    return delta;
}

auto DNN_base::backword_propagation(const base_container &expected) -> std::vector<base_container>
{
    std::vector<base_container> delta_per_layer(m_weigths_of_layers.size());
    delta_per_layer.back() = std::move(calculate_delta_of_outputs(expected));

    for (size_t layer_idx = m_weigths_of_layers.size() - 2; layer_idx > 0; --layer_idx)
    {
        const auto &next_layer_deltas = delta_per_layer[layer_idx + 1];
        const auto &cur_layer = m_weigths_of_layers[layer_idx];
        base_container deltas(cur_layer.size());

        for (size_t cur_neuron_idx = 0; cur_neuron_idx < cur_layer.size(); ++cur_neuron_idx)
        {
            const auto derivative_of_val = activation_funtions::derivative_sigmoid(cur_layer[cur_neuron_idx].get_state());
            Config::float_type error = 0.;
            size_t idx = 0;
            for (const Neuron &connected_neuron : m_weigths_of_layers[layer_idx + 1])
            {
                error = connected_neuron.get_weight(cur_neuron_idx) * next_layer_deltas[idx++];
            }
            deltas[cur_neuron_idx] = derivative_of_val * error;
        }
        delta_per_layer[layer_idx] = std::move(deltas);
    }
    return delta_per_layer;
}

void DNN_base::update_weights(const base_container &expected)
{
    const auto delta_per_layer = backword_propagation(expected);
    for (size_t i = 1; i < m_weigths_of_layers.size(); ++i)
    {
        auto &cur_layer = m_weigths_of_layers[i];
        const auto &prev_layer = m_weigths_of_layers[i - 1];
        const auto &cur_delta = delta_per_layer[i];
        size_t idx = 0;
        for (Neuron &n : cur_layer)
        {
            Config::float_type input_val = 0.;
            n.set_weight(idx, m_learning_rate * cur_delta[idx] * input_val * prev_layer[idx].get_state() + n.get_weight(idx));
            ++idx;
        }
    }
}

void DNN_base::train_network(const std::vector<std::pair<base_container, base_container>> &train)
{
    for (const auto &i : train)
    {
        initialize_input_layer(i.first);
        forward_propagation();
        update_weights(i.second);
    }
}

size_t DNN_base::predict(const base_container &input)
{
    initialize_input_layer(input);
    forward_propagation();
    const auto &outputs = m_weigths_of_layers.back();
    size_t idx = 0;
    Config::float_type max_val = outputs.front().get_state();
    for (size_t i = 1; i < outputs.size(); ++i)
    {
        if (max_val < outputs[i].get_state())
        {
            max_val = outputs[i].get_state();
            idx = i;
        }
    }
    return idx;
}