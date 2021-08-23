import argparse
import cv2
import minerl
import numpy as np
import plotly.express as px
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

def get_timeseries_reward(data_frames):
    rewards = [float(frame[2]) for frame in data_frames]
    return rewards

def get_timeseries_actions(data_frames):
    camera_null = "array([0., 0.], dtype=float32)"
    equip_null = "none"
    action_labels = sorted([key for key in data_frames[0][1].keys()]) # if key not in ["camera", "equip"]])
    actions_timeseries_wide = []
    for key in action_labels:
        if key == "camera":
            actions_timeseries_wide.append([(0 if frame[1][key] == camera_null else 1) for frame in data_frames])
        elif key == "equip":
            actions_timeseries_wide.append([(0 if frame[1][key] == equip_null else 1) for frame in data_frames])
        else:
            actions_timeseries_wide.append([float(frame[1][key]) for frame in data_frames])
    return np.array(actions_timeseries_wide), action_labels

def run_app(data_dir):
    st.set_page_config(page_title="MineRL Trajectory Viewer", page_icon=None, layout='wide')
    st.title('MineRL Trajectory Viewer')

    col1, col2, col3, col4 = st.columns([6,2,2,2])

    with col1:
        data_dir = Path(data_dir)
        st.write(f"Data dir: `{data_dir}`")

        # Select trajectory
        traj_names = get_trajectory_names(data_dir)
        option = st.selectbox(
            'Select a trajectory:',
            traj_names)
        chosen_path = data_dir / option

        env_name = str(Path(chosen_path).parent.stem)
        stream_name = Path(chosen_path).stem
        data_frames = load_data(data_dir, env_name, stream_name)

        # Select current frame
        max_frame = len(data_frames) - 1
        frame_idx = st.slider("Select frame:", 0, max_frame, 0)
        current_frame = data_frames[frame_idx]

        state, action, reward, next_state, done, meta = current_frame

        # Display GIF / video
        # st.write("## Playback")
        # st.image(str(Path(chosen_path) / "recording.mp4"))

        # Aggregate plots
        st.write("### Actions over time")
        actions_wide, action_labels = get_timeseries_actions(data_frames)
        fig = px.imshow(
            actions_wide,
            x = list(range(actions_wide.shape[1])),
            y = action_labels,
            height=300,
        )
        fig.update_traces(dict(
            showscale=False, 
            coloraxis=None,
            colorscale=[(0, "#FFF"), (1, "#3f51b5")],
            ), selector={'type':'heatmap'})
        fig.update_layout(
            margin=dict(l=0, r=20, t=20, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.write("### Rewards over time")
        rewards = get_timeseries_reward(data_frames)
        st.area_chart(rewards, height=100)

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
        st.image(state["pov"], use_column_width=True, caption="Current State POV")

        st.write("#### Equipped")
        st.write(state["equipped_items"])

        st.write("#### Inventory")
        st.write(state["inventory"])

    with col4:
        st.write("### Next State")
        st.image(next_state["pov"], use_column_width=True, caption="Current State POV")

        st.write("#### Equipped")
        st.write(next_state["equipped_items"])

        st.write("#### Inventory")
        st.write(next_state["inventory"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize MineRL trajectories')
    parser.add_argument("-d", "--data-dir", required=True,
                        help="Root directory containing trajectory data. Default: %(default)s")
    options = parser.parse_args()

    run_app(options.data_dir)