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
    'CRITIC_DISCOUNT': 0.5,
    'ENTROPY_BETA': 0.01,
    'HIDDEN_SIZE': 64,

    'LEARNING_RATE': 3e-4,
    'PPO_STEPS': 256, # Number of steps to collect in each environment per update
    'NUM_ACTORS': 2, # Number of parallel actors (environments) to collect samples from
    'MINI_BATCH_SIZE': 64, # Size of mini-batches for update
    'PPO_EPOCHS': 10, # Number of epochs to update the policy per learning cycle

    'NUM_STEPS': 30000, # Total number of steps to train the agent
    'EVAL_EPISODES': 20, # Average over this many episodes for evaluation
    'TEST_EPISODES': 25, # Number of episodes to test the agent after training

    'RENDER': RENDER,
    'DEVICE': DEVICE,
}

# --- LunarLander-v3 Configuration ---
LUNARLANDER_CONFIG = {
    'ENV_NAME': "LunarLander-v3",
    'GOAL_REWARD': 200.0,

    'GAMMA': 0.99,
    'GAE_LAMBDA': 0.95,
    'PPO_EPSILON': 0.2,
    'CRITIC_DISCOUNT': 0.5,
    'ENTROPY_BETA': 0.01,
    'HIDDEN_SIZE': 128,

    'LEARNING_RATE': 3e-4,
    'PPO_STEPS': 2048,
    'NUM_ACTORS': 8,
    'MINI_BATCH_SIZE': 64,
    'PPO_EPOCHS': 10,

    'NUM_STEPS': 1000000,
    'EVAL_EPISODES': 20,
    'TEST_EPISODES': 25,

    'RENDER': RENDER,
    'DEVICE': DEVICE,
}

# --- Acrobot-v1 Configuration ---
ACROBOT_CONFIG = {
    'ENV_NAME': "Acrobot-v1",
    'GOAL_REWARD': -100.0,

    'GAMMA': 0.99,
    'GAE_LAMBDA': 0.95,
    'PPO_EPSILON': 0.2,
    'CRITIC_DISCOUNT': 0.5,
    'ENTROPY_BETA': 0.01,
    'HIDDEN_SIZE': 64,

    'LEARNING_RATE': 3e-4,
    'PPO_STEPS': 256,
    'NUM_ACTORS': 16,
    'MINI_BATCH_SIZE': 128,
    'PPO_EPOCHS': 4,

    'NUM_STEPS': 100000,
    'EVAL_EPISODES': 20,
    'TEST_EPISODES': 25,

    'RENDER': RENDER,
    'DEVICE': DEVICE,
}

CONFIGS = {
    "CartPole-v1": CARTPOLE_CONFIG,
    "LunarLander-v3": LUNARLANDER_CONFIG,
    "Acrobot-v1": ACROBOT_CONFIG,
}
