from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.networks.q_network import QNetwork


def to_network(config, state_spec, action_spec):
    if config.class_name == 'QNetwork':
        # https://github.com/tensorflow/agents/blob/v0.7.1/tf_agents/networks/q_network.py#L43-L149
        return QNetwork(
            state_spec,
            action_spec,
            preprocessing_layers=keras.models.model_from_yaml(
                OmegaConf.to_yaml(config.networks.q_network, resolve=True)
            ).layers,
            fc_layer_params=None,
            activation_fn=None,
        )
    elif config.class_name == 'ActorDistributionNetwork':
        # https://github.com/tensorflow/agents/blob/v0.7.1/tf_agents/networks/actor_distribution_network.py#L52-L183
        return ActorDistributionNetwork(
            self.train_env.time_step_spec(),
            self.train_env.action_spec(),
            preprocessing_layers=keras.models.model_from_yaml(
                OmegaConf.to_yaml(config.networks.actor_network, resolve=True)
            ).layers,
            fc_layer_params=None,
            activation_fn=None,
           )
    elif config.class_name == 'ValueNetwork':
        # https://github.com/tensorflow/agents/blob/v0.7.1/tf_agents/networks/value_network.py#L40-L128
        return ValueNetwork(
            self.train_env.time_step_spec(),
            preprocessing_layers=keras.models.model_from_yaml(
                OmegaConf.to_yaml(config.networks.value_network, resolve=True)
            ).layers,
            fc_layer_params=None,
            activation_fn=None,
        )
    else:
        raise ValueError("Unrecognized network Class {}".format(config.class_name))
