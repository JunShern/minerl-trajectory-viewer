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
def get_timeseries_actions_fig(actions_wide, action_labels, rewards):
    st.warning("Cache miss: `get_timeseries_actions_fig` ran")
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
    return fig

class App:
    def __init__(self):
        # self.data_frames = []
        pass

    @st.cache(suppress_st_warning=True, allow_output_mutation=True, max_entries=1)
    def get_trajectory_names(self, data_dir: Path):
        st.warning("Cache miss: `get_trajectory_names` ran")
        traj_dirs = sorted([x for x in data_dir.glob("*/*") if x.is_dir()])
        traj_names = [str(Path(x.parent.stem) / x.stem) for x in traj_dirs]
        return traj_names

    @st.cache(suppress_st_warning=True, allow_output_mutation=True, max_entries=1)
    def load_data(self, data_dir, env_name, stream_name):
        st.warning("Cache miss: `load_data` ran")
        minerl_data = minerl.data.make(env_name, data_dir=data_dir)
        data_frames = list(minerl_data.load_data(stream_name, include_metadata=True))
        return data_frames

    def get_timeseries_reward(self, data_frames):
        rewards = [float(frame[2]) for frame in data_frames]
        return rewards

    def get_timeseries_actions(self, data_frames):
        action_labels = sorted([key for key in data_frames[0][1].keys()])
        actions_timeseries_wide = []
        for key in action_labels:
            action_sample = data_frames[0][1][key]
            if type(action_sample) == np.ndarray:
                actions_timeseries_wide.append([(1 if np.any(frame[1][key]) else 0) for frame in data_frames])
            elif type(action_sample) == np.str_:
                actions_timeseries_wide.append([(0 if frame[1][key] == "none" else 1) for frame in data_frames])
            elif type(action_sample) == np.int64:
                actions_timeseries_wide.append([float(frame[1][key]) for frame in data_frames])
            else:
                raise Exception(f"Action type not supported! `{action_sample}` of type `{type(action_sample)}`")
        return np.array(actions_timeseries_wide), action_labels

    def run(self, data_dir):
        st.set_page_config(page_title="MineRL Trajectory Viewer", page_icon=None, layout='wide')
        st.title('MineRL Trajectory Viewer')

        col1, col2, col3, col4 = st.columns([6,2,2,2])

        with col1:
            data_dir = Path(data_dir)
            st.write(f"Data dir: `{data_dir}`")

            # Select trajectory
            traj_names = self.get_trajectory_names(data_dir)
            option = st.selectbox(
                'Select a trajectory:',
                traj_names)
            chosen_path = data_dir / option

            env_name = str(Path(chosen_path).parent.stem)
            stream_name = Path(chosen_path).stem
            data_frames = self.load_data(data_dir, env_name, stream_name)

            # TODO: Display the video!
            # video_frames = np.stack([frame[0]["pov"].transpose(2,0,1) for frame in self.data_frames])
            # video_frames = np.stack([frame[0]["pov"] for frame in self.data_frames])

            # st.write(video_frames.shape)
            # tmp_vid_path = "vid.avi"
            # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            # video = cv2.VideoWriter(tmp_vid_path, fourcc, 20.0, video_frames[0].shape[:2])
            # for image in video_frames:
            #     video.write(image)
            # video.release()
            # print("Video saved to", tmp_vid_path)
            # st.video(tmp_vid_path)

            # # Display GIF / video
            # st.write("## Playback")
            # video_path = str(Path(chosen_path) / "recording_.mp4")
            # st.video(video_frames)
            # See Streamlit issue:
            # https://github.com/streamlit/streamlit/pull/1583

            # Select current frame
            max_frame = len(data_frames) - 1
            frame_idx = st.slider("Select frame:", 0, max_frame, 0)
            current_frame = data_frames[frame_idx]

            state, action, reward, next_state, done, meta = current_frame

            # Aggregate plots
            actions_wide, action_labels = self.get_timeseries_actions(data_frames)
            rewards = self.get_timeseries_reward(data_frames)
            fig = get_timeseries_actions_fig(actions_wide, action_labels, rewards)
            st.write("### Actions over time")
            st.plotly_chart(fig, use_container_width=True)
            st.write("### Rewards over time")
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
            for key, val in state.items():
                if key == "pov":
                    continue
                st.write(f"#### {key}")
                st.write(val)

        with col4:
            st.write("### Next State")
            st.image(next_state["pov"], use_column_width=True, caption="Next State POV")
            for key, val in next_state.items():
                if key == "pov":
                    continue
                st.write(f"#### {key}")
                st.write(val)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize MineRL trajectories')
    parser.add_argument("-d", "--data-dir", required=True,
                        help="Root directory containing trajectory data. Default: %(default)s")
    options = parser.parse_args()

    app = App()
    app.run(options.data_dir)