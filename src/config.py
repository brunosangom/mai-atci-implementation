# Configuration settings and hyperparameters
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RENDER = True # Whether to render episodes after training

# --- CartPole-v1 Configuration ---
CARTPOLE_CONFIG = {
    'ENV_NAME': "CartPole-v1",
    'GOAL_REWARD': 475.0,

    'GAMMA': 0.99,
    'GAE_LAMBDA': 0.95,
    'PPO_EPSILON': 0.2,
    'CRITIC_DISCOUNT': 0.1,
    'ENTROPY_BETA': 0.01,
    'HIDDEN_SIZE': 64,
    'NORMALIZE_ADVANTAGES': True,
    'SHARED_FEATURES': True, # Use shared features for actor and critic

    'LEARNING_RATE': 3e-4,
    'PPO_STEPS': 256, # Number of steps to collect in each environment per update
    'NUM_ACTORS': 2, # Number of parallel actors (environments) to collect samples from
    'MINI_BATCH_SIZE': 64, # Size of mini-batches for update
    'PPO_EPOCHS': 10, # Number of epochs to update the policy per learning cycle

    'NUM_STEPS': 20000, # Total number of steps to train the agent
    'EVAL_EPISODES': 0, # Average over this many episodes for evaluation
    'TEST_EPISODES': 100, # Number of episodes to test the agent after training

    'RENDER': RENDER,
    'DEVICE': DEVICE,
}

# --- HalfCheetah-v5 Configuration ---
CHEETAH_CONFIG = {
    'ENV_NAME': "HalfCheetah-v5",
    'GOAL_REWARD': 10000.0,

    'GAMMA': 0.99,
    'GAE_LAMBDA': 0.95,
    'PPO_EPSILON': 0.2,
    'CRITIC_DISCOUNT': 1,
    'ENTROPY_BETA': 0, # Do not use entropy bonus
    'HIDDEN_SIZE': 64,
    'NORMALIZE_ADVANTAGES': False,
    'SHARED_FEATURES': False,

    'LEARNING_RATE': 3e-4,
    'PPO_STEPS': 2048,
    'NUM_ACTORS': 1,
    'MINI_BATCH_SIZE': 64,
    'PPO_EPOCHS': 10,

    'NUM_STEPS': 1000000,
    'EVAL_EPISODES': 0,
    'TEST_EPISODES': 100,

    'RENDER': RENDER,
    'DEVICE': DEVICE,
}

# --- Reacher-v5 Configuration ---
REACHER_CONFIG = {
    'ENV_NAME': "Reacher-v5",
    'GOAL_REWARD': -10.0,

    'GAMMA': 0.99,
    'GAE_LAMBDA': 0.95,
    'PPO_EPSILON': 0.2,
    'CRITIC_DISCOUNT': 1,
    'ENTROPY_BETA': 0, # Do not use entropy bonus
    'HIDDEN_SIZE': 64,
    'NORMALIZE_ADVANTAGES': False,
    'SHARED_FEATURES': False,

    'LEARNING_RATE': 3e-4,
    'PPO_STEPS': 2048,
    'NUM_ACTORS': 1,
    'MINI_BATCH_SIZE': 64,
    'PPO_EPOCHS': 10,

    'NUM_STEPS': 1000000,
    'EVAL_EPISODES': 0,
    'TEST_EPISODES': 100,

    'RENDER': RENDER,
    'DEVICE': DEVICE,
}

CONFIGS = {
    "CartPole-v1": CARTPOLE_CONFIG,
    "HalfCheetah-v5": CHEETAH_CONFIG,
    "Reacher-v5": REACHER_CONFIG,
}
