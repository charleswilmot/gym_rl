import hydra
from omegaconf import OmegaConf
from procedure import Procedure
from tf_agents.environments import suite_gym


def get_action_dim(name):
    env = suite_gym.load(name)
    if env.action_spec().shape == ():
        return int(env.action_spec().maximum - env.action_spec().minimum) + 1
    else:
        return env.action_spec().shape[0]

OmegaConf.register_new_resolver("get_action_dim", get_action_dim)


@hydra.main(config_path='../conf/', config_name='defaults.yaml')
def main(cfg):
    procedure = Procedure(cfg)
    done = False
    n_steps = 0
    next_training = n_steps + cfg.training.train_every
    next_eval = n_steps + cfg.training.eval_every

    procedure.evaluate(cfg.training.n_evaluation_episodes)
    procedure.collect_data(cfg.training.n_initial_samples)
    while not done:
        todo = min(next_training, next_eval, cfg.training.n_training_steps) - n_steps
        print('collect data ({: 6d} -> {: 6d})'.format(n_steps, n_steps + todo), end='\r')
        procedure.collect_data(todo)
        n_steps += todo
        must_train = n_steps == next_training
        must_eval = n_steps == next_eval
        done = n_steps == cfg.training.n_training_steps
        if must_train:
            loss = procedure.train_from_buffer(cfg.training.train_every * cfg.training.n_training_per_sample)
            print('loss: {}'.format(loss))
            next_training = n_steps + cfg.training.train_every
        if must_eval:
            procedure.evaluate(cfg.training.n_evaluation_episodes)
            next_eval = n_steps + cfg.training.eval_every

    if not must_eval:
        procedure.evaluate(cfg.training.n_evaluation_episodes)


if __name__ == '__main__':
    main()
