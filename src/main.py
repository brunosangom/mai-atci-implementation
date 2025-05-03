from train import train_agent
from config import CONFIGS

# --- Environment Selection ---
ENVIRONMENTS = ["CartPole-v1", "Acrobot-v1", "MountainCarContinuous-v0"]
SELECTED_ENV = ENVIRONMENTS[0]

# --- Main Execution ---
if __name__ == "__main__":
    # Update the config with the selected environment
    current_config = CONFIGS[SELECTED_ENV]

    train_agent(current_config)
