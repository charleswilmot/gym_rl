import tensorflow as tf
from tfagents_sequential import TFAgentSequential


def dummy_activation_fn(x):
    return x

DEFAULT_LAYER_SIZE = 123
ACTION_DIM = 234
ACTIVATION_FN = dummy_activation_fn


def post_process(string):
    return string.replace(
        'dummy_activation_fn', '${networks.default_activation_fn}'
    ).replace(
        '123', '${networks.default_layer_size}'
    ).replace(
        '234', '${environment.action_dim}'
    )


for n_layers in [2, 3, 4]:
    for outdim in ['n_actions', 'one']:
        for activation in ['linear', 'tanh']:
            net = TFAgentSequential([
                tf.keras.layers.Dense(
                    DEFAULT_LAYER_SIZE,
                    activation=ACTIVATION_FN,
                    kernel_initializer=tf.keras.initializers.VarianceScaling(
                        scale=2.0,
                        mode='fan_in',
                        distribution='truncated_normal')
                ) for i in range(n_layers - 1)] + [
                tf.keras.layers.Dense(
                    ACTION_DIM if outdim == 'n_actions' else 1,
                    activation=None if activation == 'linear' else tf.tanh,
                    kernel_initializer=tf.keras.initializers.VarianceScaling(
                        scale=2.0,
                        mode='fan_in',
                        distribution='truncated_normal')
                ),
            ])

            with open('../conf/networks/{}_layers_outdim_{}_{}.yaml'.format(n_layers, outdim, activation), 'w') as f:
                f.write(post_process(net.to_yaml()))


# net = tf.keras.models.model_from_yaml()
