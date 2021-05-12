python experiment.py environment.name=CartPole-v0 training.eval_every=500 training.n_evaluation_episodes=10 training.n_training_steps=5000
python experiment.py environment.name=LunarLander-v2 training.eval_every=500 training.n_evaluation_episodes=2 rl_algorithm.agent.config.gamma=0.99 rl_algorithm.agent.config.target_update_tau=0.01 training.n_training_steps=120000
# python experiment.py environment.name=Pendulum-v0 rl_algorithm=ppo training.eval_every=500 training.n_evaluation_episodes=10 training.n_training_steps=5000 training.train_every=250
