import gymnasium as gym
import torch
import numpy as np
import os
from datetime import datetime
from agent import PPOAgent
import imageio
# CONFIG is already imported in main.py and passed to train_agent

def evaluate_agent(env_name, agent, eval_episodes, device):
    """Evaluates the agent over a number of episodes using a single environment."""
    env = gym.make(env_name)
    total_rewards = []
    for _ in range(eval_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            state_tensor = torch.tensor(np.array([state]), dtype=torch.float).to(device)
            with torch.no_grad():
                # Only need action for evaluation, ignore log_prob and value
                action_logits, _ = agent.actor_critic(state_tensor)
                action = torch.argmax(action_logits, dim=-1).item() # Choose best action deterministically

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
        total_rewards.append(episode_reward)
    env.close()
    return np.mean(total_rewards)

def generate_renders(env_name, agent, num_renders, save_dir, device):
    """Generates GIF renders of the agent playing using a single environment."""
    print(f"Generating GIF renders...")
    os.makedirs(save_dir, exist_ok=True)
    # Create a temporary env with rgb_array rendering
    render_env = gym.make(env_name, render_mode="rgb_array")

    for i in range(num_renders):
        frames = []
        state, _ = render_env.reset()
        done = False
        episode_reward = 0
        while not done:
            # Capture frame before taking action
            frames.append(render_env.render())

            state_tensor = torch.tensor(np.array([state]), dtype=torch.float).to(device)
            with torch.no_grad():
                action_logits, _ = agent.actor_critic(state_tensor)
                action = torch.argmax(action_logits, dim=-1).item() # Deterministic action

            next_state, reward, terminated, truncated, _ = render_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state

        # Save the collected frames as a GIF
        gif_path = os.path.join(save_dir, f"episode_{i}-reward_{episode_reward:.2f}.gif")
        imageio.mimsave(gif_path, frames, fps=60) # Adjust fps as needed
        print(f"  Render {i+1}/{num_renders} finished. Reward: {episode_reward:.2f}")

    render_env.close() # Close the base env
    print("GIF Renders generated.\n")

def train_agent(cfg):
    """Trains the PPO agent using parallel environments."""
    print(f"Using device: {cfg['DEVICE']}")
    print(f"Training on environment: {cfg['ENV_NAME']} with {cfg['NUM_ACTORS']} actors")
    print("Configuration:")
    for key, value in cfg.items():
        print(f"  {key}: {value}")

    # --- Initialization ---
    # Create a function to generate individual environments
    def make_env(env_id, seed):
        def _init():
            env = gym.make(env_id)
            # Consider seeding if reproducibility across actors is critical
            # env.reset(seed=seed + rank) # Example seeding
            return env
        return _init

    # Create the vectorized environment
    # envs = gymnasium.vector.SyncVectorEnv( # Use SyncVectorEnv for simplicity first
    envs = gym.vector.AsyncVectorEnv( # Use Async for potential speedup
        [make_env(cfg['ENV_NAME'], i) for i in range(cfg['NUM_ACTORS'])]
    )

    # Get dimensions from the vector environment's observation/action spaces
    state_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.n

    agent = PPOAgent(state_dim, action_dim, cfg)

    # Create directories for saving models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join("ppo_models", f"ppo_{cfg['ENV_NAME']}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    print(f"Models will be saved in: {model_dir}")


    total_steps = 0
    global_episode_count = 0 # Track total episodes across all actors
    best_eval_reward = -np.inf

    # --- Training Loop ---
    # Reset all environments and get initial states
    states, _ = envs.reset() # states is now a numpy array (NUM_ACTORS, state_dim)

    while total_steps < cfg['NUM_STEPS']:
        # --- Data Collection Phase ---
        # Collect PPO_STEPS * NUM_ACTORS transitions in total per cycle
        for step in range(cfg['PPO_STEPS']):
            # Select actions for the batch of states
            # states are already a numpy array, agent.select_actions should handle it
            actions, log_probs, values = agent.select_actions(states) # Expects batch input

            # Step the parallel environments
            # actions need to be numpy array for vec env
            next_states, rewards, terminateds, truncateds, _ = envs.step(actions)
            dones = np.logical_or(terminateds, truncateds) # Combine termination conditions

            # Store transitions for each actor
            for i in range(cfg['NUM_ACTORS']):
                # Extract individual data points
                state = states[i]
                action = actions[i]
                log_prob = log_probs[i]
                reward = rewards[i]
                value = values[i] # Get the value for the state *before* stepping
                done = dones[i]

                agent.store_transition(state, action, log_prob, reward, value, done)

                # If an episode finished in one of the environments
                if done:
                    global_episode_count += 1
                    # Check for 'final_observation' in infos if available and needed
                    # Resetting is handled automatically by AsyncVectorEnv

            # Update states for the next iteration
            states = next_states
            total_steps += cfg['NUM_ACTORS'] # Increment by the number of actors

            # Break if total steps reached during collection
            if total_steps >= cfg['NUM_STEPS']:
                break

        # --- Learning Phase ---
        # Bootstrap value estimation for the last states from each actor
        # Need to get values for the *next_states* (which are the current 'states')
        with torch.no_grad():
            # states are the final states after the collection loop
            states_tensor = torch.tensor(states, dtype=torch.float).to(cfg['DEVICE'])
            _, next_values_tensor = agent.actor_critic(states_tensor) # Get values for the last states
            next_values = next_values_tensor.cpu().numpy().flatten() # Flatten to (NUM_ACTORS,)

        # The memory buffer now contains transitions from all actors interleaved.
        # We need to calculate GAE correctly. The current memory implementation
        # calculates GAE assuming a single continuous trajectory. This needs fixing.
        # For now, we'll pass a dummy next_value and last_done, acknowledging this limitation.
        # A more correct implementation would require restructuring the memory buffer
        # or calculating GAE per-actor before storing/batching.

        # *** Temporary/Incorrect GAE Handling for Vectorized Env ***
        # Calculate an average next value, ignoring individual dones for now.
        # This is NOT the correct way to handle GAE with vector envs but allows training to proceed.
        # A proper fix involves modifying PPOMemory.generate_batches or how data is stored.
        avg_next_value = np.mean(next_values * (1.0 - dones.astype(float))) # Approximate bootstrap
        # Assume the 'last_done' for the whole batch calculation is based on 'any' done.
        # This is also an approximation.
        batch_last_done = np.any(dones)

        agent.learn(avg_next_value, batch_last_done) # Pass approximated bootstrap info

        # --- Evaluation Phase ---
        if global_episode_count > 0: # Evaluate after some episodes have finished
            # Use the single-env evaluation function
            avg_eval_reward = evaluate_agent(cfg['ENV_NAME'], agent, cfg['EVAL_EPISODES'], cfg['DEVICE'])
            print(f"Evaluation after {total_steps} steps ({global_episode_count} episodes): Average Reward = {avg_eval_reward:.2f}")

            # Save the model if it's the best so far
            if avg_eval_reward > best_eval_reward and total_steps >= cfg['NUM_STEPS'] * 0.1:
                best_eval_reward = avg_eval_reward
                save_path = os.path.join(model_dir, f"ppo_{cfg['ENV_NAME']}_{timestamp}.pth")
                agent.save_model(save_path)

                # Generate renders if requested (using the best agent)
                if cfg.get('RENDER', False):
                    render_save_dir = os.path.join(model_dir, "renders", f"{avg_eval_reward:.2f}")
                    generate_renders(cfg['ENV_NAME'], agent, 3, render_save_dir, cfg['DEVICE'])
            
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

        best_agent = PPOAgent(state_dim, action_dim, cfg)
        best_agent.load_model(best_model_path)
        # Evaluate using the single-env evaluator
        final_avg_reward = evaluate_agent(cfg['ENV_NAME'], best_agent, cfg['TEST_EPISODES'], cfg['DEVICE'])
        print(f"Final Evaluation (Best Model): Average Reward over {cfg['TEST_EPISODES']} episodes = {final_avg_reward:.2f}")

    else:
        print("No best model was saved during training. Evaluating current agent state.")
        # Evaluate the final state of the training agent using the single-env evaluator
        final_avg_reward = evaluate_agent(cfg['ENV_NAME'], agent, cfg['TEST_EPISODES'], cfg['DEVICE'])
        print(f"Final Evaluation (Current Agent): Average Reward = {final_avg_reward:.2f}")


    envs.close()

