#include <cassert>
#include <future>
#include <type_traits>
#include <vector>

#include "activation_functions.h"
#include "base_neuron.h"

class DNN_base
{
private:
    using base_container = std::vector<Config::float_type>;

public:
    DNN_base(const std::vector<unsigned>& layers_size);

    void initialize_input_layer(const base_container states) noexcept(Config::DBG_BUILD);

    Config::float_type calculate_activation(unsigned layer_idx, unsigned neuron_idx) noexcept(Config::DBG_BUILD);

    void forward_propagation() noexcept(Config::DBG_BUILD);

    base_container calculate_delta_of_outputs(const base_container &expected) const noexcept(Config::DBG_BUILD);

    std::vector<base_container> backword_propagation(const base_container &expected);

    void update_weights(const base_container &expected);

    void train_network(const std::vector<std::pair<base_container, base_container>> &train);

    size_t predict(const base_container &input);

private:
    std::vector<std::vector<Neuron>> m_weigths_of_layers;
    Config::float_type m_learning_rate;
};