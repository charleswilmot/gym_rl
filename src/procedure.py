from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.networks.q_network import QNetwork
from tf_agents.environments import suite_gym,tf_py_environment
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents import agents
from tf_agents import replay_buffers
from tensorflow import keras
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
import tensorflow as tf


def get_optimizer(optimizer_config):
    optimizer_class_name = optimizer_config.class_name
    if optimizer_class_name == 'Adam':
        return keras.optimizers.Adam(**optimizer_config.config)
    else:
        raise ValueError("Unrecognized optimizer class name ({})".format(optimizer_class_name))



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
            q_network = QNetwork(
                self.train_env.observation_spec(),
                self.train_env.action_spec(),
                preprocessing_layers=keras.models.model_from_yaml(
                    OmegaConf.to_yaml(config.networks.q_network, resolve=True)
                ),
                fc_layer_params=None,
                activation_fn=None,
            )
            target_q_network = q_network.copy()
            td_errors_loss_fn = common.element_wise_squared_loss if config.rl_algorithm.agent.config.td_errors_loss_fn == 'squared' else None
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
                td_errors_loss_fn=common.element_wise_squared_loss,
                train_step_counter=self.train_step_counter,
            )
        elif agent_class_name == 'PPOAgent':
            actor_net = ActorDistributionNetwork(
                self.train_env.observation_spec(),
                self.train_env.action_spec(),
                preprocessing_layers=keras.models.model_from_yaml(
                    OmegaConf.to_yaml(config.networks.actor_network, resolve=True)
                ),
                fc_layer_params=None,
                activation_fn=None,
            )
            value_net = ValueNetwork(
                self.train_env.observation_spec(),
                preprocessing_layers=keras.models.model_from_yaml(
                    OmegaConf.to_yaml(config.networks.value_network, resolve=True)
                ),
                fc_layer_params=None,
                activation_fn=None,
            )
            self.agent = agents.ppo.ppo_agent.PPOAgent(
                self.train_env.time_step_spec(),
                self.train_env.action_spec(),
                optimizer=get_optimizer(config.rl_algorithm.agent.config.optimizer),
                actor_net=actor_net,
                value_net=value_net,
                importance_ratio_clipping=config.rl_algorithm.agent.config.importance_ratio_clipping,
                lambda_value=config.rl_algorithm.agent.config.lambda_value,
                discount_factor=config.rl_algorithm.agent.config.discount_factor,
                entropy_regularization=config.rl_algorithm.agent.config.entropy_regularization,
                policy_l2_reg=config.rl_algorithm.agent.config.policy_l2_reg,
                value_function_l2_reg=config.rl_algorithm.agent.config.value_function_l2_reg,
                shared_vars_l2_reg=config.rl_algorithm.agent.config.shared_vars_l2_reg,
                value_pred_loss_coef=config.rl_algorithm.agent.config.value_pred_loss_coef,
                num_epochs=config.rl_algorithm.agent.config.num_epochs,
                use_gae=config.rl_algorithm.agent.config.use_gae,
                use_td_lambda_return=config.rl_algorithm.agent.config.use_td_lambda_return,
                normalize_rewards=config.rl_algorithm.agent.config.normalize_rewards,
                reward_norm_clipping=config.rl_algorithm.agent.config.reward_norm_clipping,
                normalize_observations=config.rl_algorithm.agent.config.normalize_observations,
                log_prob_clipping=config.rl_algorithm.agent.config.log_prob_clipping,
                kl_cutoff_factor=config.rl_algorithm.agent.config.kl_cutoff_factor,
                kl_cutoff_coef=config.rl_algorithm.agent.config.kl_cutoff_coef,
                initial_adaptive_kl_beta=config.rl_algorithm.agent.config.initial_adaptive_kl_beta,
                adaptive_kl_target=config.rl_algorithm.agent.config.adaptive_kl_target,
                adaptive_kl_tolerance=config.rl_algorithm.agent.config.adaptive_kl_tolerance,
                gradient_clipping=config.rl_algorithm.agent.config.gradient_clipping,
                value_clipping=config.rl_algorithm.agent.config.value_clipping,
                train_step_counter=self.train_step_counter,
            )
        else:
            raise ValueError("Unrecognized agent class name ({})".format(agent_class_name))
        self.agent.initialize()

        # - instantiate a buffer object
        buffer_class_name = config.rl_algorithm.buffer.class_name
        if buffer_class_name == 'TFUniformReplayBuffer':
            self.replay_buffer = replay_buffers.tf_uniform_replay_buffer.TFUniformReplayBuffer(
                data_spec=self.agent.collect_data_spec,
                batch_size=self.train_env.batch_size,
                max_length=config.rl_algorithm.buffer.config.max_length)
        else:
            raise ValueError("Unrecognized buffer class name ({})".format(buffer_class_name))
        self.dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=config.training.batch_size,
            num_steps=2).prefetch(3)
        self.iterator = iter(self.dataset)

        # tensorboard logging
        self.summary_writer = tf.summary.create_file_writer('./logs')
        self.acc_train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.step_counter = 0
        self.log_freq = config.training.log_freq

    def collect_data_one_step(self):
        # - collect data --> into buffer
        time_step = self.train_env.current_time_step()
        action_step = self.agent.collect_policy.action(time_step)
        next_time_step = self.train_env.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        self.replay_buffer.add_batch(traj)

    def collect_data(self, n_steps):
        for i in range(n_steps):
            self.collect_data_one_step()

    # @tf.function
    def train_from_buffer_once(self):
        # - sample data from buffer --> for training
        experience, unused_info = next(self.iterator)
        loss = self.agent.train(experience).loss
        self.acc_train_loss(loss)
        self.step_counter += 1
        if self.step_counter % self.log_freq == 0:
            with self.summary_writer.as_default():
                tf.summary.scalar('train_loss', self.acc_train_loss.result(), step=self.step_counter)
                self.acc_train_loss.reset_states()
        return loss

    def train_from_buffer(self, n_times):
        loss = 0
        for i in range(n_times):
            loss += self.train_from_buffer_once()
        return loss / n_times

    def evaluate(self, n_episodes):
        # - if some condition is met --> evaluate the perf of the agent
        #       - this data must be logged in tensorboard / into the SQL database
        total_return = 0.0
        for _ in range(n_episodes):
            time_step = self.eval_env.reset()
            episode_return = 0.0
            while not time_step.is_last():
                action_step = self.agent.policy.action(time_step)
                time_step = self.eval_env.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return
        avg_return = total_return / n_episodes
        print('evaluation: {}'.format(avg_return.numpy()[0]))
        with self.summary_writer.as_default():
            tf.summary.scalar('return', avg_return.numpy()[0], step=self.step_counter)
        return avg_return.numpy()[0]


    # def record_video(self):
    #   - record a video of the trained agent
