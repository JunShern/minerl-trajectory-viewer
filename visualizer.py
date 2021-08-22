import cv2
import numpy as np
import streamlit as st
import time
from pathlib import Path

# # Initialization
# if 'first_run' not in st.session_state:
#     st.session_state['first_run'] = False
#     st.session_state['frame_num'] = 1
#     st.session_state['is_playing'] = False

st.title('MineRL Trajectory Viewer')

# Select trajectory
data_dir = Path("output")
img_dirs = sorted([str(x) for x in data_dir.glob("*/*") if x.is_dir()])
option = st.selectbox(
    'Select a trajectory:',
     img_dirs)
    
# Display GIF / video
st.write("## Playback")
st.image(str(Path(option).with_suffix(".gif")))

# Frame-by-frame analysis
st.write("## Frame-by-frame analysis")
img_paths = sorted(list(Path(option).glob("*.jpg")))
max_frame = len(img_paths) - 1


frame_num = st.slider("Select frame:", 0, max_frame, 0)
img = cv2.imread(str(img_paths[frame_num]))
st.image(img, caption=f"`{img_paths[frame_num]}`")

# st.write(st.session_state)

# We clear elements by calling empty on them.
# progress_bar.empty()
# frame_text.empty()

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
# st.button("Re-run")