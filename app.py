import streamlit as st
import cv2
import tempfile

# Page configuration
st.set_page_config(
    page_title="SRWPUO - Real-Time Webcam Processing",
    page_icon="📸",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    </style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">Real-Time Webcam Processing</h1>', unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.header("Controls")

    # Camera selection
    camera_option = st.radio(
        "Select Camera Source:",
        ("Webcam", "Upload Video File")
    )

    # Processing options
    st.subheader("Processing Options")

    processing_options = st.multiselect(
        "Select processing effects:",
        ["Grayscale", "Edge Detection", "Blur", "Canny Edge", "Cartoon Effect", "Invert Colors"]
    )

    # Blur intensity
    blur_intensity = st.slider("Blur Intensity", 1, 15, 5, 1,
                               disabled="Blur" not in processing_options)

    # Canny thresholds
    if "Canny Edge" in processing_options:
        col1, col2 = st.columns(2)
        with col1:
            canny_low = st.slider("Canny Low", 50, 200, 100, 10)
        with col2:
            canny_high = st.slider("Canny High", 100, 400, 200, 10)

    # Frame rate control
    fps = st.slider("Frame Rate (FPS)", 1, 30, 15)

    # Start/Stop button
    start_processing = st.button("Start Processing")
    stop_processing = st.button("Stop Processing")

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("Live Feed")
    frame_placeholder = st.empty()

with col2:
    st.subheader("Processed Output")
    processed_placeholder = st.empty()

# Status indicator
status_text = st.empty()

# File uploader for video files
uploaded_file = None
if camera_option == "Upload Video File":
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])


# Image processing functions
def process_frame(frame, options, blur_intensity=5, canny_low=100, canny_high=200):
    """Apply selected processing effects to frame"""
    processed = frame.copy()

    for option in options:
        if option == "Grayscale":
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        elif option == "Edge Detection":
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
            processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        elif option == "Blur":
            processed = cv2.GaussianBlur(processed, (blur_intensity * 2 + 1, blur_intensity * 2 + 1), 0)

        elif option == "Canny Edge":
            if len(processed.shape) == 3:
                gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            else:
                gray = processed
            edges = cv2.Canny(gray, canny_low, canny_high)
            processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        elif option == "Cartoon Effect":
            # Convert to grayscale
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

            # Apply median blur
            gray = cv2.medianBlur(gray, 5)

            # Detect edges
            edges = cv2.adaptiveThreshold(gray, 255,
                                          cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, 9, 9)

            # Color quantization
            color = cv2.bilateralFilter(processed, 9, 250, 250)
            cartoon = cv2.bitwise_and(color, color, mask=edges)
            processed = cartoon

        elif option == "Invert Colors":
            processed = cv2.bitwise_not(processed)

    return processed


# Video processing function
def process_video():
    cap = None

    try:
        if camera_option == "Webcam":
            cap = cv2.VideoCapture(0)
        elif camera_option == "Upload Video File" and uploaded_file is not None:
            # Save uploaded file to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)

        if cap and cap.isOpened():
            status_text.success("✅ Processing started!")

            while start_processing and not stop_processing:
                ret, frame = cap.read()

                if not ret:
                    status_text.warning("Cannot read frame. Trying to restart...")
                    break

                # Process the frame
                processed_frame = process_frame(
                    frame,
                    processing_options,
                    blur_intensity,
                    canny_low if "Canny Edge" in processing_options else 100,
                    canny_high if "Canny Edge" in processing_options else 200
                )

                # Convert frames for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                # Display frames
                frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                processed_placeholder.image(processed_rgb, channels="RGB", use_column_width=True)

                # Control frame rate
                cv2.waitKey(int(1000 / fps))

    except Exception as e:
        status_text.error(f"❌ Error: {str(e)}")

    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        status_text.info("Processing stopped")


# Instructions section
with st.expander("📖 How to Use"):
    st.markdown("""
    1. **Select Camera Source** in the sidebar:
        - Use your webcam
        - Or upload a video file

    2. **Choose Processing Effects**:
        - Grayscale: Convert to black and white
        - Edge Detection: Highlight edges
        - Blur: Apply Gaussian blur (adjust intensity)
        - Canny Edge: Advanced edge detection
        - Cartoon Effect: Apply cartoon-style filter
        - Invert Colors: Invert color palette

    3. **Adjust Settings**:
        - Blur intensity (if blur is selected)
        - Canny thresholds (if Canny Edge is selected)
        - Frame rate (FPS)

    4. **Click 'Start Processing'** to begin
    5. **Click 'Stop Processing'** to end
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>SRWPUO - Streamlit Real-Time Webcam Processing Using OpenCV</p>
    <p>Developed with 🎔 using Streamlit and OpenCV</p>
</div>
""", unsafe_allow_html=True)

# Run processing if start button is clicked
if start_processing:
    process_video()