import argparse
import cv2
import minerl
import numpy as np
import random
import streamlit as st
from pathlib import Path

def run_app(data_dir):
    st.title('MineRL Trajectory Viewer')

    # Select trajectory
    data_dir = Path(data_dir)
    traj_dirs = sorted([x for x in data_dir.glob("*/*") if x.is_dir()])
    traj_names = [str(Path(x.parent.stem) / x.stem) for x in traj_dirs]
    option = st.selectbox(
        'Select a trajectory:',
        traj_names)
    chosen_path = data_dir / option

    env_name = str(Path(chosen_path).parent.stem)
    stream_name = Path(chosen_path).stem
    minerl_data = minerl.data.make(env_name, data_dir=data_dir)

    data_frames = list(minerl_data.load_data(stream_name, include_metadata=True))
    
    # Display GIF / video
    # st.write("## Playback")
    # st.image(str(Path(chosen_path) / "recording.mp4"))

    # Frame-by-frame analysis
    st.write("## Frame-by-frame analysis")
    max_frame = len(data_frames) - 1
    frame_idx = st.slider("Select frame:", 0, max_frame, 0)

    img = data_frames[frame_idx][0]["pov"]
    st.image(img)

    meta = data_frames[frame_idx][-1]
    st.write("Metadata:", meta)
    st.write(data_frames[frame_idx])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize MineRL trajectories')
    parser.add_argument("-d", "--data-dir", required=True,
                        help="Root directory containing trajectory data. Default: %(default)s")
    options = parser.parse_args()

    run_app(options.data_dir)