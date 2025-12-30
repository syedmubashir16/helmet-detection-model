import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

# Set page configuration
st.set_page_config(
    page_title="Helmet Detection System",
    page_icon="👷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve the UI
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
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the YOLO model."""
    try:
        # Check relative to the script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'best.pt')
        
        if not os.path.exists(model_path):
             # Fallback to current working directory
             if os.path.exists('best.pt'):
                 model_path = 'best.pt'
             else:
                curr_dir = os.getcwd()
                files_in_dir = os.listdir(curr_dir)
                script_dir_files = os.listdir(script_dir)
                st.error(f"Model file not found. Checked: {model_path} and 'best.pt'")
                st.error(f"Current Directory: {curr_dir}")
                st.error(f"Files in Current Directory: {files_in_dir}")
                st.error(f"Script Directory: {script_dir}")
                st.error(f"Files in Script Directory: {script_dir_files}")
                return None
        
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    # Sidebar
    st.sidebar.title("Configuration")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.25, 
        step=0.05
    )
    
    st.title("👷 Helmet Detection System")
    st.markdown("### Upload an image to detect hard hats and safety violations.")
    
    # Load Model
    model = load_model()
    
    if model:
        # File Uploader
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "webp"])
        
        if uploaded_file is not None:
            # Display original image
            col1, col2 = st.columns(2)
            
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption='Original Image', use_container_width=True)
            
            # Perform detection
            with col2:
                with st.spinner('Detecting...'):
                    # Convert PIL image to format compatible with YOLO (if needed, but YOLO handles PIL)
                    # Run inference
                    results = model.predict(image, conf=confidence_threshold)
                    
                    # Visualize the results
                    for result in results:
                        # plot() returns a numpy array in BGR
                        res_plotted = result.plot()
                        # Convert BGR to RGB for Streamlit/PIL
                        res_rgb = res_plotted[..., ::-1]
                        st.image(res_rgb, caption='Detected Image', use_container_width=True)
                    
                    # Show detection stats
                    st.success("Detection Complete!")
                    
                    # Optional: Display detected classes and counts
                    if results:
                        result = results[0]
                        boxes = result.boxes
                        if len(boxes) > 0:
                            class_names = result.names
                            # Count instances
                            classes = boxes.cls.cpu().numpy()
                            unique, counts = np.unique(classes, return_counts=True) 
                            class_counts = dict(zip(unique, counts))
                            
                            st.write("### Detection Summary:")
                            stats_df = []
                            for cls_id, count in class_counts.items():
                                name = class_names[int(cls_id)]
                                stats_df.append({"Object Type": name, "Count": count})
                            
                            st.table(stats_df)
                        else:
                            st.warning("No objects detected.")

if __name__ == "__main__":
    main()
