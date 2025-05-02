from train import train_agent
from config import CONFIG

# --- Environment Selection ---
ENVIRONMENTS = ["CartPole-v1", "LunarLander-v2", "Acrobot-v1"]
GOAL_REWARDS = {
    "CartPole-v1": 475.0,
    "LunarLander-v2": 200.0,
    "Acrobot-v1": -100.0,
}
SELECTED_ENV = ENVIRONMENTS[0]

# --- Main Execution ---
if __name__ == "__main__":
    # Update the config with the selected environment
    current_config = CONFIG.copy()
    current_config['ENV_NAME'] = SELECTED_ENV
    current_config['GOAL_REWARD'] = GOAL_REWARDS[SELECTED_ENV]

    train_agent(current_config)
