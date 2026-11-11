import streamlit as st
import cv2
import os
import tempfile
from ultralytics import YOLO

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="YOLOv8 Vision System",
    page_icon="🎯",
    layout="wide" 
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
.main-title {
    text-align: center;
    color: #00A6FF;
    font-size: 38px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: gray;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown(
    '<p class="main-title">YOLOv8 Vision System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">AI-Powered Computer Vision Application</p>',
    unsafe_allow_html=True
)

st.divider()

# ---------------- SIDEBAR ----------------

st.sidebar.title("🎯 YOLO Applications")

task = st.sidebar.selectbox(
    "Select Application",
    [
        "Object Detection",
        "Object Counting",
        "Object Segmentation",
        "Object Tracking",
        "Pose Estimation"
    ]
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.10,
    1.00,
    0.25
)

st.sidebar.divider()

st.sidebar.info(
    "YOLOv8 Multi-Task Computer Vision Project"
)

# ---------------- MODEL SELECTION ----------------

models = {
    "Object Detection": "yolov8n.pt",
    "Object Counting": "yolov8n.pt",
    "Object Segmentation": "yolov8n-seg.pt",
    "Object Tracking": "yolov8n.pt",
    "Pose Estimation": "yolov8n-pose.pt"
}

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)


# ---------------- APPLICATION DETAILS ----------------

st.subheader(f"📌 {task}")

descriptions = {
    "Object Detection":
        "Detect objects using bounding boxes.",

    "Object Counting":
        "Detect and count objects in video frames.",

    "Object Segmentation":
        "Identify objects using pixel-level masks.",

    "Object Tracking":
        "Track detected objects with unique IDs.",

    "Pose Estimation":
        "Detect human body keypoints and skeletons."
}

st.write(descriptions[task])

# ---------------- VIDEO UPLOAD ----------------

uploaded_file = st.file_uploader(
    "Upload your video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_file is not None:

    st.video(uploaded_file)

    if st.button("▶ Start Processing", type="primary"):

        model_path = models[task]

        # Temporary directory for input and output
        with tempfile.TemporaryDirectory() as temp_dir:

            input_path = os.path.join(
                temp_dir,
                "input_video" + os.path.splitext(uploaded_file.name)[1]
            )

            output_path = os.path.join(
                temp_dir,
                "output_video.mp4"
            )

            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:

                model = load_model(model_path)

                cap = cv2.VideoCapture(input_path)

                if not cap.isOpened():
                    st.error("Unable to open video.")
                    st.stop()

                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 25

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                out = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    fps,
                    (width, height)
                )

                if not out.isOpened():
                    cap.release()
                    st.error("Unable to create output video.")
                    st.stop()

                progress = st.progress(0)
                status = st.empty()
                preview = st.empty()

                total_frames = int(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT)
                )

                frame_count = 0
                total_detections = 0

                while True:

                    ret, frame = cap.read()

                    if not ret:
                        break

                    # -------- YOLO INFERENCE --------

                    if task == "Object Tracking":

                        results = model.track(
                            frame,
                            persist=True,
                            tracker="bytetrack.yaml",
                            conf=confidence,
                            verbose=False
                        )

                    else:

                        results = model.predict(
                            frame,
                            conf=confidence,
                            verbose=False
                        )

                    # -------- ANNOTATION --------

                    annotated_frame = results[0].plot()

                    detections = (
                        len(results[0].boxes)
                        if results[0].boxes is not None
                        else 0
                    )

                    total_detections += detections

                    # -------- OBJECT COUNTING --------

                    if task == "Object Counting":

                        cv2.putText(
                            annotated_frame,
                            f"Object Count: {detections}",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )

                    # -------- POSE ESTIMATION --------

                    if task == "Pose Estimation":

                        cv2.putText(
                            annotated_frame,
                            f"Persons: {detections}",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )

                    # -------- SAVE FRAME --------

                    out.write(annotated_frame)

                    frame_count += 1

                    # -------- LIVE PREVIEW --------

                    if frame_count == 1 or frame_count % 10 == 0:

                        preview.image(
                            cv2.cvtColor(
                                annotated_frame,
                                cv2.COLOR_BGR2RGB
                            ),
                            channels="RGB"
                        )

                    # -------- PROGRESS --------

                    if total_frames > 0:

                        progress.progress(
                            min(frame_count / total_frames, 1.0)
                        )

                    status.write(
                        f"Processing frame: {frame_count}"
                    )

                cap.release()
                out.release()

                if frame_count == 0:
                    st.error("No frames found in the video.")
                    st.stop()

                st.success("Processing completed successfully!")

                # -------- RESULTS --------

                col1, col2 = st.columns(2)

                col1.metric(
                    "Frames Processed",
                    frame_count
                )

                col2.metric(
                    "Total Detections",
                    total_detections
                )

                # -------- OUTPUT VIDEO --------

                st.subheader("🎬 Processed Video")

                with open(output_path, "rb") as f:
                    video_bytes = f.read()

                st.video(video_bytes)

                st.download_button(
                    "⬇ Download Processed Video",
                    data=video_bytes,
                    file_name="yolov8_output.mp4",
                    mime="video/mp4"
                )

            except Exception as e:

                st.error(f"Error: {e}")

else:

    st.info("Please upload a video to start the application.")

# ---------------- FOOTER ----------------

st.divider()

st.markdown(
    "<center>Developed using Python | Streamlit | "
    "OpenCV | YOLOv8</center>",
    unsafe_allow_html=True
)
