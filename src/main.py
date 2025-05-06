from train import train_agent
from config import CONFIGS

# --- Environment Selection ---
ENVIRONMENTS = ["CartPole-v1", "HalfCheetah-v5", "Reacher-v5"]
SELECTED_ENV = ENVIRONMENTS[2]

# --- Main Execution ---
if __name__ == "__main__":
    # Update the config with the selected environment
    current_config = CONFIGS[SELECTED_ENV]

    train_agent(current_config)