import argparse
import cv2
import minerl
import numpy as np
import random
import streamlit as st
from pathlib import Path

import time

@st.cache(suppress_st_warning=True, allow_output_mutation=True, max_entries=1)
def get_trajectory_names(data_dir: Path):
    st.warning("Cache miss: `get_trajectory_names` ran")
    traj_dirs = sorted([x for x in data_dir.glob("*/*") if x.is_dir()])
    traj_names = [str(Path(x.parent.stem) / x.stem) for x in traj_dirs]
    return traj_names

@st.cache(suppress_st_warning=True, allow_output_mutation=True, max_entries=1)
def load_data(data_dir, env_name, stream_name):
    st.warning("Cache miss: `load_data` ran")
    minerl_data = minerl.data.make(env_name, data_dir=data_dir)
    data_frames = list(minerl_data.load_data(stream_name, include_metadata=True))
    return data_frames

def run_app(data_dir):
    st.set_page_config(page_title="MineRL Trajectory Viewer", page_icon=None, layout='wide')
    st.title('MineRL Trajectory Viewer')

    # col1, col2 = st.columns(2)
    col1, col2, col3 = st.columns([2,1,1])

    with col1:
        # Select trajectory
        data_dir = Path(data_dir)
        st.write(f"Root data directory: `{data_dir}`")
        traj_names = get_trajectory_names(data_dir)
        option = st.selectbox(
            'Select a trajectory:',
            traj_names)
        chosen_path = data_dir / option

        env_name = str(Path(chosen_path).parent.stem)
        stream_name = Path(chosen_path).stem
        data_frames = load_data(data_dir, env_name, stream_name)

        # Display GIF / video
        # st.write("## Playback")
        # st.image(str(Path(chosen_path) / "recording.mp4"))

        # Select current frame
        max_frame = len(data_frames) - 1
        frame_idx = st.slider("Select frame:", 0, max_frame, 0)
        current_frame = data_frames[frame_idx]

        state, action, reward, next_state, done, meta = current_frame

        st.image(state["pov"], width=300, caption="State POV")
        # st.image(next_state["pov"], width=300, caption="Next state POV")

    with col2:
        st.write("### Actions")
        st.write(action)
        st.write("### Reward")
        st.write(f"`{reward}`")
        st.write("### Done")
        st.write(done)
        st.write("### Metadata")
        st.write(meta)

    with col3:
        st.write("### State")
        st.write("#### Equipped")
        st.write(state["equipped_items"])

        st.write("#### Inventory")
        st.write(state["inventory"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize MineRL trajectories')
    parser.add_argument("-d", "--data-dir", required=True,
                        help="Root directory containing trajectory data. Default: %(default)s")
    options = parser.parse_args()

    run_app(options.data_dir)