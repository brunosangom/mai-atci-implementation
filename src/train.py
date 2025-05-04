import gymnasium as gym
import torch
import numpy as np
import os
from datetime import datetime
from agent import PPOAgent
from gymnasium.wrappers import RecordVideo, ClipAction
import glob
import json

def evaluate_agent(env_name, agent, eval_episodes, device, is_continuous):
    """Evaluates the agent over a number of episodes using a single environment."""
    env = gym.make(env_name)
    if is_continuous:
        env = ClipAction(env) # Clip actions to the environment's bounds

    total_rewards = []
    for _ in range(eval_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            state_tensor = torch.tensor(np.array([state]), dtype=torch.float).to(device)
            with torch.no_grad():
                # Get deterministic action from the agent's model
                action_tensor = agent.actor_critic.get_deterministic_action(state_tensor)
                action = action_tensor.cpu().numpy()
                action = action[0] if is_continuous else action.item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
        total_rewards.append(episode_reward)
    env.close()
    return np.mean(total_rewards)

def generate_renders(env_name, agent, num_renders, save_dir, device, is_continuous):
    """Generates video renders of the agent playing using RecordVideo."""
    print(f"Generating video renders...")
    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_renders):
        # Create a new RecordVideo wrapped env for each render
        temp_video_folder = os.path.join(save_dir, f"temp_render_{i}")
        render_env_base = gym.make(env_name, render_mode="rgb_array")
        if is_continuous:
            render_env_base = ClipAction(render_env_base) # Clip actions

        render_env = RecordVideo(
            render_env_base,
            video_folder=temp_video_folder,
            episode_trigger=lambda x: True, # Record every episode
            name_prefix=f"rl-video-render-{i}" # Unique prefix for this render
        )

        state, _ = render_env.reset()
        done = False
        episode_reward = 0
        while not done:
            state_tensor = torch.tensor(np.array([state]), dtype=torch.float).to(device)
            with torch.no_grad():
                # Get deterministic action
                action_tensor = agent.actor_critic.get_deterministic_action(state_tensor)
                action = action_tensor.cpu().numpy()
                action = action[0] if is_continuous else action.item()

            next_state, reward, terminated, truncated, _ = render_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state

        # Closing the env triggers the video saving by RecordVideo
        render_env.close()

        # --- Rename the generated video file ---
        video_files = glob.glob(os.path.join(temp_video_folder, "*.mp4"))
        if video_files:
            source_path = video_files[0]
            # Define the desired destination path with the reward included in the filename
            destination_path = os.path.join(save_dir, f"episode_{i}-reward_{episode_reward:.2f}.mp4")
            # Rename the file
            os.rename(source_path, destination_path)
            # Remove the temporary directory
            os.rmdir(temp_video_folder)
            print(f"  Render {i+1}/{num_renders} finished. Reward: {episode_reward:.2f}")
        else:
             print(f"  Render {i+1}/{num_renders} finished. Reward: {episode_reward:.2f}. Error: Could not find saved video file in {temp_video_folder}.")


    print("Video Renders generated.\n")

def train_agent(cfg):
    """Trains the PPO agent using parallel environments."""
    print(f"Using device: {cfg['DEVICE']}")
    print(f"Training on environment: {cfg['ENV_NAME']} with {cfg['NUM_ACTORS']} actors")
    print("Configuration:")
    for key, value in cfg.items():
        print(f"  {key}: {value}")

    # --- Initialization ---
    # Create a temporary single env to get space info
    temp_env = gym.make(cfg['ENV_NAME'])
    is_continuous = isinstance(temp_env.action_space, gym.spaces.Box)
    state_dim = temp_env.observation_space.shape[0]

    if is_continuous:
        action_dim = temp_env.action_space.shape[0]
        action_low = temp_env.action_space.low
        action_high = temp_env.action_space.high
        print(f"Detected Continuous action space (dim={action_dim}, low={action_low}, high={action_high})")
    else:
        action_dim = temp_env.action_space.n
        action_low, action_high = None, None # Not applicable for discrete
        print(f"Detected Discrete action space (n={action_dim})")
    temp_env.close() # Close the temporary environment

    # Create a function to generate individual environments
    def make_env(env_id):
        def _init():
            env = gym.make(env_id)
            # Apply ClipAction wrapper for continuous environments within the vector env
            if is_continuous:
                env = ClipAction(env)
            return env
        return _init

    # Create the vectorized environment to allow parallel actors
    envs = gym.vector.AsyncVectorEnv(
        [make_env(cfg['ENV_NAME']) for _ in range(cfg['NUM_ACTORS'])]
    )

    agent = PPOAgent(state_dim, action_dim, cfg, is_continuous=is_continuous)

    # Create directories for saving models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join("ppo_models", f"ppo_{cfg['ENV_NAME']}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    print(f"Models, config, and renders will be saved in: {model_dir}")

    # Save the configuration dictionary as a JSON file
    config_path = os.path.join(model_dir, "config.json")
    try:
        # Add action space info to config before saving
        cfg_to_save = cfg.copy()
        cfg_to_save['is_continuous'] = is_continuous
        cfg_to_save['action_dim'] = int(action_dim)
        if is_continuous:
            cfg_to_save['action_low'] = action_low.tolist() # Convert numpy arrays for JSON
            cfg_to_save['action_high'] = action_high.tolist()

        with open(config_path, 'w') as f:
            json.dump(cfg_to_save, f, indent=4)
        print(f"Configuration saved to {config_path}")
    except Exception as e:
        print(f"Error saving configuration to {config_path}: {e}")

    total_steps = 0
    global_episode_count = 0 # Track total episodes across all actors
    best_eval_reward = -np.inf

    # --- Training Loop ---
    # Reset all environments and get initial states
    states, _ = envs.reset()

    while total_steps < cfg['NUM_STEPS']:
        # --- Data Collection Phase ---
        # Collect PPO_STEPS * NUM_ACTORS transitions in total per cycle
        for step in range(cfg['PPO_STEPS']):
            # Select actions for the batch of states
            actions, log_probs, values = agent.select_actions(states)

            # Step the parallel environments
            next_states, rewards, terminateds, truncateds, _ = envs.step(actions)
            dones = np.logical_or(terminateds, truncateds) # Combine termination conditions

            # Store transitions for each actor
            for i in range(cfg['NUM_ACTORS']):
                # Extract individual data points
                state = states[i]
                action = actions[i]
                log_prob = log_probs[i]
                reward = rewards[i]
                value = values[i]
                done = dones[i]

                agent.store_transition(state, action, log_prob, reward, value, done)

                # If an episode finished in one of the environments
                if done:
                    global_episode_count += 1
                    # Resetting the env is handled automatically by AsyncVectorEnv

            # Update states for the next iteration
            states = next_states
            total_steps += cfg['NUM_ACTORS'] # Increment by the number of actors

        # --- Learning Phase ---
        # Bootstrap value estimation for the last states from each actor
        with torch.no_grad():
            # states are the final states after the collection loop
            states_tensor = torch.tensor(states, dtype=torch.float).to(cfg['DEVICE'])
            # Get values for the last states reached by each actor
            _, next_values_tensor = agent.actor_critic(states_tensor)
            # Ensure next_values is a flat numpy array (NUM_ACTORS,)
            next_values = next_values_tensor.squeeze().cpu().numpy()
            # Ensure dones is a numpy boolean array (NUM_ACTORS,)
            last_dones = dones.astype(bool)


        # Pass the actual next values and done flags for each actor to learn
        agent.learn(next_values, last_dones)

        # --- Evaluation Phase ---
        if global_episode_count > 0: # Evaluate after some episodes have finished
            # Use the single-env evaluation function
            avg_eval_reward = evaluate_agent(cfg['ENV_NAME'], agent, cfg['EVAL_EPISODES'], cfg['DEVICE'], is_continuous)
            print(f"Evaluation after {total_steps} steps ({global_episode_count} episodes): Average Reward = {avg_eval_reward:.2f}")

            # Save the model if it's the best so far
            if avg_eval_reward > best_eval_reward and (total_steps >= cfg['NUM_STEPS'] * 0.1 or avg_eval_reward >= cfg['GOAL_REWARD']):
                best_eval_reward = avg_eval_reward
                save_path = os.path.join(model_dir, f"ppo_{cfg['ENV_NAME']}_{timestamp}.pth")
                agent.save_model(save_path)

                # Generate renders if requested
                if cfg.get('RENDER', False):
                    render_save_dir = os.path.join(model_dir, "renders", f"{avg_eval_reward:.3f}")
                    generate_renders(cfg['ENV_NAME'], agent, 3, render_save_dir, cfg['DEVICE'], is_continuous)
            
            # Check goal reward
            if best_eval_reward >= cfg['GOAL_REWARD']:
                print(f"Goal reward of {cfg['GOAL_REWARD']} reached! Stopping training.")
                break


    # --- Final Evaluation & Saving ---
    print("\nTraining complete.")
    # Load the best model for final evaluation
    best_model_path = os.path.join(model_dir, f"ppo_{cfg['ENV_NAME']}_{timestamp}.pth")
    if os.path.exists(best_model_path):
        print(f"Loading best model for final evaluation.")

        best_agent = PPOAgent(state_dim, action_dim, cfg, is_continuous=is_continuous)
        best_agent.load_model(best_model_path)
        # Evaluate using the single-env evaluator
        final_avg_reward = evaluate_agent(cfg['ENV_NAME'], best_agent, cfg['TEST_EPISODES'], cfg['DEVICE'], is_continuous)
        print(f"Final Evaluation (Best Model): Average Reward over {cfg['TEST_EPISODES']} episodes = {final_avg_reward:.2f}")

    else:
        print("No best model was saved during training. Evaluating current agent state.")
        # Evaluate the final state of the training agent using the single-env evaluator
        final_avg_reward = evaluate_agent(cfg['ENV_NAME'], agent, cfg['TEST_EPISODES'], cfg['DEVICE'], is_continuous)
        print(f"Final Evaluation (Current Agent): Average Reward = {final_avg_reward:.2f}")


    envs.close()

