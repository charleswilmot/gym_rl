from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents import agents
from tf_agents import replay_buffers


def get_optimizer(optimizer_config):
    pass


def to_network(network_config):
    pass


class Procedure:
    def __init__(self, config):

        # - instantiate a gym environment and wrap it in a TFEnvironment
        train_py_env = suite_gym.load(config.environment.name)
        eval_py_env = suite_gym.load(config.environment.name)
        self.train_env = tf_py_environment.TFPyEnvironment(train_py_env)
        self.eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

        # - instantiate an agent (RL algo)
        self.train_step_counter = tf.Variable(0)
        agent_class_name = config.rl_algorithm.agent.class_name
        if agent_class_name == 'DqnAgent':
            q_network = to_network(config.rl_algorithm.agent.config.q_network)
            target_q_network = to_network(config.rl_algorithm.agent.config.q_network)
            self.agent = agents.dqn.dqn_agent.DqnAgent(
                self.train_env.time_step_spec(),
                self.train_env.action_spec(),
                q_network=q_network,
                target_q_network=target_q_network,
                optimizer=get_optimizer(config.rl_algorithm.agent.config.optimizer),
                epsilon_greedy=config.rl_algorithm.agent.config.epsilon_greedy,
                n_step_update=config.rl_algorithm.agent.config.n_step_update,
                boltzmann_temperature=config.rl_algorithm.agent.config.boltzmann_temperature,
                target_update_tau=config.rl_algorithm.agent.config.target_update_tau,
                target_update_period=config.rl_algorithm.agent.config.target_update_period,
                gamma=config.rl_algorithm.agent.config.gamma,
                reward_scale_factor=config.rl_algorithm.agent.config.reward_scale_factor,
                train_step_counter=self.train_step_counter,
            )
        elif agent_class_name == 'PPOAgent':
            self.agent = agents.ppo.PPOAgent(
            )
        else:
            raise ValueError("Unrecognized agent class name ({})".format(agent_class_name))

        # - instantiate a buffer object
        buffer_class_name = config.rl_algorithm.buffer.class_name
        if buffer_class_name == 'TFUniformReplayBuffer':
            self.replay_buffer = replay_buffers.tf_uniform_replay_buffer.TFUniformReplayBuffer(
                data_spec=self.agent.collect_data_spec,
                batch_size=self.train_env.batch_size,
                max_length=config.rl_algorithm.buffer.config.max_length)
        elif buffer_class_name == 'SomethingElse':
            pass
        else:
            raise ValueError("Unrecognized buffer class name ({})".format(buffer_class_name))

    def initialize_buffer(self):
      - initialize the buffer with some data

    def collect_data(self):
      - collect data --> into buffer

    def train_from_buffer(self):
      - sample data from buffer --> for training

    def evaluate(self):
      - if some condition is met --> evaluate the perf of the agent
            - this data must be logged in tensorboard / into the SQL database

    def record_video(self):
      - record a video of the trained agent
