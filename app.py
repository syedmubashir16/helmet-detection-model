import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Streamlit page configuration
st.set_page_config(
    page_title="Helmet Detection System",
    page_icon="👷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache the model so it loads only once
@st.cache_resource
def load_model():
    """Load YOLO model from best.pt in project root."""
    try:
        model = YOLO("best.pt")   # Simplified path for Streamlit Cloud
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    # Sidebar configuration
    st.sidebar.title("Configuration")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05
    )

    # Main title
    st.title("👷 Helmet Detection System")
    st.markdown("### Upload an image to detect hard hats and safety violations.")

    # Load YOLO model
    model = load_model()

    if model:
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png", "bmp", "webp"]
        )

        if uploaded_file is not None:
            # Display original image
            col1, col2 = st.columns(2)

            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption="Original Image", use_container_width=True)

            # Perform detection
            with col2:
                with st.spinner("Detecting..."):
                    results = model.predict(image, conf=confidence_threshold)

                    for result in results:
                        # Plot detection results (BGR numpy array)
                        res_plotted = result.plot()
                        res_rgb = res_plotted[..., ::-1]  # Convert BGR → RGB
                        st.image(res_rgb, caption="Detected Image", use_container_width=True)

                    st.success("Detection Complete!")

                    # Detection summary
                    if results:
                        result = results[0]
                        boxes = result.boxes
                        if len(boxes) > 0:
                            class_names = result.names
                            classes = boxes.cls.cpu().numpy()
                            unique, counts = np.unique(classes, return_counts=True)
                            class_counts = dict(zip(unique, counts))

                            st.write("### Detection Summary:")
                            stats = []
                            for cls_id, count in class_counts.items():
                                name = class_names[int(cls_id)]
                                stats.append({"Object Type": name, "Count": count})

                            st.table(stats)
                        else:
                            st.warning("No objects detected.")

if __name__ == "__main__":
    main()