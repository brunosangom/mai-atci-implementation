import streamlit as st
import os

# Attempt to set MUJOCO_GL to osmesa for headless rendering if it's not already set.
# This is a potential workaround for GLFW errors on headless servers if OSMesa is available.
# It should be set before gymnasium or mujoco is imported if they auto-initialize GL.
if 'MUJOCO_GL' not in os.environ:
    print("Streamlit App: Attempting to set MUJOCO_GL=osmesa for headless MuJoCo rendering.")
    os.environ['MUJOCO_GL'] = 'osmesa'

import sys
import glob
import json
from datetime import datetime
import torch
import gymnasium as gym
import numpy as np

# Adjust sys.path to import from src
# This assumes 'streamlit' and 'src' are sibling directories under 'mai-atci-implementation'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.config import CONFIGS  # To get environment names
    from src.agent import PPOAgent
    from gymnasium.wrappers import RecordVideo, ClipAction
except ImportError as e:
    st.error(f"Failed to import necessary modules from 'src'. Ensure 'src' is in the Python path: {e}")
    st.stop()


# Base directory for models, relative to this script's parent directory
MODELS_BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'ppo_models')
# Base directory for Streamlit renders, relative to this script's directory
STREAMLIT_RENDERS_DIR = os.path.join(os.path.dirname(__file__), 'streamlit_renders')
os.makedirs(STREAMLIT_RENDERS_DIR, exist_ok=True)


def find_latest_model_dir(env_name):
    """Finds the directory of the latest model for the given environment."""
    pattern = os.path.join(MODELS_BASE_DIR, f"ppo_{env_name}_*")
    matching_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]

    if not matching_dirs:
        return None

    def get_timestamp_from_dir(dir_path):
        dir_name = os.path.basename(dir_path)
        parts = dir_name.split('_')
        # Expecting format like ppo_ENV-NAME_YYYYMMDD_HHMMSS
        # Timestamp is formed by the last two parts
        if len(parts) >= 3: # Need at least ppo, name, date, time
            return parts[-2] + "_" + parts[-1]
        return "0" # Fallback for sorting if format is unexpected

    try:
        latest_dir = sorted(matching_dirs, key=get_timestamp_from_dir, reverse=True)[0]
        return latest_dir
    except IndexError: # Should not happen if matching_dirs is not empty
        return None


def load_agent_from_model_dir(model_dir_path):
    """Loads the agent and its configuration from the model directory."""
    config_path = os.path.join(model_dir_path, "config.json")
    model_pth_files = glob.glob(os.path.join(model_dir_path, "*.pth"))

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {model_dir_path}")
    if not model_pth_files:
        raise FileNotFoundError(f"No .pth model file found in {model_dir_path}")

    model_pth_path = model_pth_files[0]

    with open(config_path, 'r') as f:
        agent_config = json.load(f)

    # These parameters are crucial and should be in the saved config.json
    state_dim = agent_config['state_dim']
    action_dim = agent_config['action_dim']
    is_continuous = agent_config['is_continuous']
    
    # Ensure DEVICE is sensible for Streamlit context, fallback to CPU if not available
    if 'DEVICE' not in agent_config or (agent_config['DEVICE'] == 'cuda' and not torch.cuda.is_available()):
        # st.warning(f"Device in config was {agent_config.get('DEVICE')}, but falling back to CPU.")
        agent_config['DEVICE'] = "cpu"
    
    # PPOAgent constructor uses the full agent_config dictionary.
    # It internally initializes ActorCritic which uses HIDDEN_SIZE from this config.
    agent = PPOAgent(state_dim, action_dim, agent_config, is_continuous=is_continuous)
    agent.load_model(model_pth_path)
    agent.actor_critic.eval()  # Set to evaluation mode

    return agent, agent_config


def generate_streamlit_render(env_name, agent, agent_config, save_dir_base):
    """Generates a single video render for Streamlit display."""
    device = agent_config['DEVICE']
    is_continuous = agent_config['is_continuous']

    os.makedirs(save_dir_base, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    render_episode_folder = os.path.join(save_dir_base, f"render_{env_name.replace('-','_')}_{timestamp}")
    os.makedirs(render_episode_folder, exist_ok=True)

    render_env = None
    render_env_base = None
    video_path = None

    try:
        render_env_base = gym.make(env_name, render_mode="rgb_array")
        if is_continuous:
            render_env_base = ClipAction(render_env_base)

        render_env = RecordVideo(
            render_env_base,
            video_folder=render_episode_folder,
            episode_trigger=lambda x: x == 0,  # Record only the first episode
            name_prefix=f"streamlit-render-{env_name.replace('-','_')}"
        )

        state, _ = render_env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            state_np = np.array([state], dtype=np.float32) if not isinstance(state, np.ndarray) or state.ndim == 1 else state
            state_tensor = torch.tensor(state_np, dtype=torch.float).to(device)
            
            with torch.no_grad():
                action_tensor = agent.actor_critic.get_deterministic_action(state_tensor)
                action = action_tensor.cpu().numpy()
                action = action[0] if is_continuous else action.item() # Adjust for batch dim if present

            next_state, reward, terminated, truncated, _ = render_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
        
        # Video is saved when render_env is closed or episode ends based on trigger
        # For RecordVideo, closing is essential to finalize.
        render_env.close() 

        video_files = glob.glob(os.path.join(render_episode_folder, "*.mp4"))
        if video_files:
            original_video_path = video_files[0]
            # Optional: Rename for clarity, though st.video doesn't require it.
            new_video_name = f"episode_0-reward_{episode_reward:.2f}.mp4"
            final_video_path = os.path.join(render_episode_folder, new_video_name)
            if original_video_path != final_video_path : # only rename if names are different
                 if os.path.exists(final_video_path): # remove if exists to avoid error on windows
                     os.remove(final_video_path)
                 os.rename(original_video_path, final_video_path)
            video_path = final_video_path
            st.write(f"Episode finished. Reward: {episode_reward:.2f}")
        else:
            st.error(f"Could not find saved video file in {render_episode_folder}.")

    except Exception as e:
        st.error(f"Error during render generation: {e}")
        import traceback
        st.text(traceback.format_exc())
    finally:
        if render_env: # RecordVideo wrapper
            render_env.close()
        if render_env_base: # Original env
            render_env_base.close()
            
    return video_path

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("PPO Model Visualization")

# Check if src modules were loaded
if 'CONFIGS' not in globals() or 'PPOAgent' not in globals():
    st.error("Core components from 'src' directory could not be loaded. Please check the setup.")
    st.stop()

env_names = list(CONFIGS.keys())
if not env_names:
    st.error("No environments found in CONFIGS from src.config.py")
    st.stop()

selected_env = st.selectbox("Choose an Environment", env_names)

if st.button("Generate and Visualize Episode", key="generate_button"):
    if selected_env:
        with st.spinner(f"Processing for {selected_env}..."):
            # st.info(f"Searching for the latest model for {selected_env} in {MODELS_BASE_DIR}...")
            latest_model_dir = find_latest_model_dir(selected_env)

            if latest_model_dir:
                # st.success(f"Found model directory: {os.path.basename(latest_model_dir)}")
                try:
                    agent, agent_config = load_agent_from_model_dir(latest_model_dir)
                    st.info(f"Agent loaded successfully using device: {agent_config['DEVICE']}. Generating video render...")
                    
                    video_path = generate_streamlit_render(
                        env_name=selected_env,
                        agent=agent,
                        agent_config=agent_config,
                        save_dir_base=STREAMLIT_RENDERS_DIR
                    )

                    if video_path and os.path.exists(video_path):
                        # Use columns to control video width
                        # Make the video column take 60% of the width, centered
                        col1, col2, col3 = st.columns([0.3, 0.4, 0.3]) 
                        with col2:
                            st.video(video_path)
                    elif video_path: # Path returned but file doesn't exist
                        st.error(f"Video generation reported path {video_path}, but file not found.")
                    else: # No video path returned
                        st.error("Failed to generate or find the video.")
                
                except FileNotFoundError as fnf_error:
                    st.error(f"File not found during agent loading: {fnf_error}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    import traceback
                    st.text(traceback.format_exc()) # Show full traceback in app for easier debugging
            else:
                st.error(f"No trained models found for {selected_env} in {MODELS_BASE_DIR}. Pattern: ppo_{selected_env}_*")
    else:
        st.warning("Please select an environment.")

st.markdown("---")
# st.markdown(f"Models are loaded from: `{MODELS_BASE_DIR}`")
# st.markdown(f"Renders are saved to: `{STREAMLIT_RENDERS_DIR}`")

# To run this Streamlit app:
# 1. Ensure you are in the 'mai-atci-implementation' directory.
# 2. Run the command: streamlit run streamlit/app.py
