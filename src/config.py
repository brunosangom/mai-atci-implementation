# Configuration settings and hyperparameters
import torch

# Environment
# ENV_NAME = "CartPole-v1" # Removed: Will be set in main.py
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
LEARNING_RATE = 3e-4

GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPSILON = 0.2
CRITIC_DISCOUNT = 0.5
ENTROPY_BETA = 0.01
HIDDEN_SIZE = 64

PPO_STEPS = 256 # Number of steps to collect in each environment per update
NUM_ACTORS = 2 # Number of parallel actors (environments) to collect samples from
MINI_BATCH_SIZE = 64 # Size of mini-batches for update
PPO_EPOCHS = 10 # Number of epochs to update the policy per learning cycle

GOAL_REWARD = 475.0 # Reward threshold for CartPole-v1
NUM_STEPS = 30000 # Total number of steps to train the agent
EVAL_EPISODES = 20 # Average over this many episodes for evaluation
TEST_EPISODES = 25 # Number of episodes to test the agent after training

RENDER = True # Whether to render episodes after training

CONFIG = {
    # 'ENV_NAME': ENV_NAME, # Removed: Set in main.py
    'DEVICE': DEVICE,
    'LEARNING_RATE': LEARNING_RATE,
    'GAMMA': GAMMA,
    'GAE_LAMBDA': GAE_LAMBDA,
    'PPO_EPSILON': PPO_EPSILON,
    'CRITIC_DISCOUNT': CRITIC_DISCOUNT,
    'ENTROPY_BETA': ENTROPY_BETA,
    'HIDDEN_SIZE': HIDDEN_SIZE,
    'PPO_STEPS': PPO_STEPS,
    'NUM_ACTORS': NUM_ACTORS,
    'MINI_BATCH_SIZE': MINI_BATCH_SIZE,
    'PPO_EPOCHS': PPO_EPOCHS,
    'GOAL_REWARD': GOAL_REWARD,
    'NUM_STEPS': NUM_STEPS,
    'EVAL_EPISODES': EVAL_EPISODES,
    'TEST_EPISODES': TEST_EPISODES,
    'RENDER': RENDER,
}
