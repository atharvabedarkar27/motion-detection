import cv2
import streamlit as st
import numpy as np
from PIL import Image

st.title("Motion Detection and Marking")

mode = st.radio("Select input source:", ("Webcam", "Upload Video"))
frame_placeholder = st.empty()
save_motion = st.checkbox("Save frames with motion", value=False)
min_area = st.slider("Minimum contour area for motion", min_value=1, max_value=1000, value=20, step=1)

if mode == "Upload Video":
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if video_file is not None:
        tfile = open("temp_video.mp4", "wb")
        tfile.write(video_file.read())
        tfile.close()
        cap = cv2.VideoCapture("temp_video.mp4")
        ret, prev_frame = cap.read()
        if not ret:
            st.error("Could not read video file.")
        else:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            frame_count = 0
            play = st.button("Play Video", key="play_video_btn")
            while play and cap.isOpened():
                ret, current_frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % 3 != 0:
                    continue
                current_frame = cv2.resize(current_frame, (640, 360))
                current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(prev_gray, current_gray)
                _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                motion_detected = False
                for contour in contours:
                    if cv2.contourArea(contour) < min_area:
                        continue
                    motion_detected = True
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(current_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                prev_gray = current_gray
                frame_count += 1
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB")
                if motion_detected and save_motion:
                    cv2.imwrite(f"motion_frame_{frame_count}.jpg", current_frame)
            cap.release()

elif mode == "Webcam":
    if 'webcam_running' not in st.session_state:
        st.session_state['webcam_running'] = False

    start_col, stop_col = st.columns(2)
    with start_col:
        if st.button("Start Webcam", key="start_webcam_btn"):
            st.session_state['webcam_running'] = True
    with stop_col:
        if st.button("Stop Webcam", key="stop_webcam_btn"):
            st.session_state['webcam_running'] = False

    if st.session_state['webcam_running']:
        cap = cv2.VideoCapture(0)
        ret, prev_frame = cap.read()
        if not ret:
            st.error("Could not read from webcam.")
            st.session_state['webcam_running'] = False
        else:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            frame_count = 0
            while cap.isOpened() and st.session_state['webcam_running']:
                ret, current_frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % 3 != 0:
                    continue
                current_frame = cv2.resize(current_frame, (640, 360))
                current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(prev_gray, current_gray)
                _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                motion_detected = False
                for contour in contours:
                    if cv2.contourArea(contour) < min_area:
                        continue
                    motion_detected = True
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(current_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                prev_gray = current_gray
                frame_count += 1
                frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB")
                if motion_detected and save_motion:
                    cv2.imwrite(f"motion_frame_{frame_count}.jpg", current_frame)
                # Allow Streamlit to process UI events
                if not st.session_state['webcam_running']:
                    break
            cap.release()