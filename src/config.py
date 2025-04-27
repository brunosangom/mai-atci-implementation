\
# Configuration settings and hyperparameters

# Environment
ENV_NAME = "CartPole-v1"

# Training
LEARNING_RATE = 3e-4
GAMMA = 0.99  # Discount factor
GAE_LAMBDA = 0.95 # Lambda for Generalized Advantage Estimation
PPO_EPSILON = 0.2 # PPO clip parameter
CRITIC_DISCOUNT = 0.5 # Critic loss coefficient
ENTROPY_BETA = 0.01 # Entropy coefficient
PPO_STEPS = 2048 # Number of steps per PPO update
MINI_BATCH_SIZE = 64
PPO_EPOCHS = 10 # Number of epochs per PPO update
TEST_EPOCHS = 10 # Number of episodes to test agent performance
TARGET_REWARD = 500 # Target reward to stop training
MAX_FRAMES = 100000 # Maximum number of frames (steps) to train for

# Network
HIDDEN_SIZE = 256

# Device
DEVICE = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"
