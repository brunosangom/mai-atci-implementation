from train import train_agent
from config import CONFIG

# --- Environment Selection ---
ENVIRONMENTS = ["CartPole-v1", "LunarLander-v2", "Acrobot-v1"] # Add more environments here
SELECTED_ENV = ENVIRONMENTS[0] # Select the environment (e.g., the first one)

# --- Main Execution ---
if __name__ == "__main__":
    # Update the config with the selected environment
    current_config = CONFIG.copy()
    current_config['ENV_NAME'] = SELECTED_ENV

    train_agent(current_config)
