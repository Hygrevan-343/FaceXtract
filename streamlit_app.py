import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import uuid
import zipfile
from typing import List, Tuple, Dict
from mtcnn import MTCNN
import time
from concurrent.futures import ThreadPoolExecutor

# Set page config
st.set_page_config(
    page_title="FaceXtract | ASAI",
    page_icon="👥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS with Google Font
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@600&display=swap');
    
    .main .block-container {
        max-width: 800px;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    .title {
        font-family: 'Fredoka', sans-serif;
        text-align: center;
        font-size: 3.5rem !important;
        font-weight: 600;
        margin: 2rem 0 !important;
        padding: 0.5rem;
        background: linear-gradient(90deg, #FF4B6C, #FF6B6B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.2);
        letter-spacing: 2px;
    }
    
    .upload-container {
        background: rgba(17, 25, 40, 0.25);
        border: 2px dashed rgba(255, 75, 108, 0.5);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-container:hover {
        border-color: #ff4b6c;
        background: rgba(255, 75, 108, 0.1);
    }
    
    .stButton > button {
        width: 100%;
        padding: 0.75rem;
        border-radius: 10px;
        background: linear-gradient(90deg, #ff4b6c, #ff6b6b);
        color: white;
        border: none;
        font-weight: 600;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 75, 108, 0.2);
    }
    
    .uploaded-files {
        background: rgba(17, 25, 40, 0.25);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .file-item {
        display: flex;
        align-items: center;
        padding: 0.5rem;
        margin: 0.5rem 0;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
    }
    
    .stats-container {
        background: rgba(17, 25, 40, 0.25);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #ff4b6c, #ff6b6b);
        border-radius: 10px;
    }
    
    .stAlert {
        border-radius: 10px;
        border-left-color: #ff4b6c;
    }
    
    .stImage {
        margin: 1rem 0;
        border-radius: 10px;
        overflow: hidden;
    }
    
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }
        .title {
            font-size: 2.5rem !important;
        }
    }
</style>

<h1 class="title">FaceXtract</h1>
""", unsafe_allow_html=True)

# Initialize MTCNN detector with caching
@st.cache_resource
def load_detector():
    return MTCNN(min_face_size=20)

detector = load_detector()

def format_size(size_bytes):
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} GB"

def process_single_image(args):
    """Process a single image with face detection"""
    image, confidence_threshold = args
    detections = detector.detect_faces(image)
    return [det for det in detections if det['confidence'] >= confidence_threshold]

def convert_to_cv2(uploaded_file) -> np.ndarray:
    """Convert uploaded file to CV2 format."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def draw_faces(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw bounding boxes around detected faces."""
    image_copy = image.copy()
    for det in detections:
        x, y, w, h = det['box']
        confidence = det['confidence']
        
        # Draw rectangle with bright green color and thicker line
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Draw confidence score with better visibility
        label = f"{confidence:.2f}"
        labelSize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        y_label = max(y - 10, labelSize[1])
        
        # Draw black background for text for better contrast
        cv2.rectangle(image_copy, 
                     (x - 1, y_label - labelSize[1] - 5),
                     (x + labelSize[0] + 1, y_label + 5),
                     (0, 0, 0),
                     cv2.FILLED)
        
        # Draw text in bright green
        cv2.putText(image_copy,
                    label,
                    (x, y_label),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,  # Larger font size
                    (0, 255, 0),  # Bright green color
                    2)  # Thicker text
        
        # Draw a small dot at each corner of the box for better visibility
        dot_size = 4
        # Top-left
        cv2.circle(image_copy, (x, y), dot_size, (0, 255, 0), -1)
        # Top-right
        cv2.circle(image_copy, (x + w, y), dot_size, (0, 255, 0), -1)
        # Bottom-left
        cv2.circle(image_copy, (x, y + h), dot_size, (0, 255, 0), -1)
        # Bottom-right
        cv2.circle(image_copy, (x + w, y + h), dot_size, (0, 255, 0), -1)
    
    return image_copy

def extract_face(image: np.ndarray, detection: Dict) -> Image.Image:
    """Extract and resize face region."""
    x, y, w, h = detection['box']
    
    # Ensure coordinates are within bounds
    x = max(0, x)
    y = max(0, y)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)
    
    # Extract face region
    face = image[y:y+h, x:x+w]
    
    # Resize with padding to 80x80
    face_pil = Image.fromarray(face)
    return face_pil.resize((80, 80), Image.Resampling.LANCZOS)

def create_zip_file(faces_by_image: Dict[str, List[Image.Image]]) -> bytes:
    """Create ZIP file with extracted faces."""
    if not faces_by_image:
        return None
        
    zip_buffer = io.BytesIO()
    total_faces = 0
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for img_idx, faces in faces_by_image.items():
            folder_name = f'image_{img_idx}'
            for face in faces:
                face_filename = f'{uuid.uuid4().hex[:6]}.png'
                face_buffer = io.BytesIO()
                face.save(face_buffer, format='PNG')
                face_buffer.seek(0)
                zip_path = f'{folder_name}/{face_filename}'
                zip_file.writestr(zip_path, face_buffer.getvalue())
                total_faces += 1
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue() if total_faces > 0 else None

def process_images(uploaded_files, confidence_threshold: float) -> Tuple[bytes, float]:
    """Process uploaded images and return ZIP file and processing time."""
    start_time = time.time()
    faces_by_image = {}
    total_processed = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Convert all images to CV2 format first
    images = []
    for uploaded_file in uploaded_files:
        image = convert_to_cv2(uploaded_file)
        uploaded_file.seek(0)
        images.append(image)
    
    # Process images in parallel
    with ThreadPoolExecutor() as executor:
        args = [(img, confidence_threshold) for img in images]
        results = list(executor.map(process_single_image, args))
    
    # Process results and create visualizations
    for idx, (image, detections) in enumerate(zip(images, results), 1):
        status_text.text(f"Processing Image {idx}/{len(uploaded_files)}")
        
        if detections:
            image_with_faces = draw_faces(image, detections)
            st.image(image_with_faces, 
                    caption=f"Detected {len(detections)} faces",
                    use_column_width=True)
            
            faces = [extract_face(image, det) for det in detections]
            faces_by_image[str(idx)] = faces
            total_processed += len(faces)
        else:
            st.warning(f"No faces detected in image {idx}")
            st.image(image, caption="No faces detected", use_column_width=True)
        
        progress_bar.progress((idx) / len(uploaded_files))
    
    processing_time = time.time() - start_time
    status_text.text(f"Processing Complete! Found {total_processed} faces in {processing_time:.1f} seconds")
    
    return create_zip_file(faces_by_image), processing_time

def main():
    # Description text
    st.markdown('<p style="text-align: center; color: #718096; margin-bottom: 1rem;">UPLOAD PHOTOS TO EXTRACT INDIVIDUAL FACE IMAGES</p>', unsafe_allow_html=True)

    # Confidence threshold slider
    confidence_threshold = st.slider(
        "Detection Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.15,
        step=0.05,
        help="Lower values will detect more faces but may include false positives"
    )

    # File uploader
    uploaded_files = st.file_uploader(
        "Drop your Images here",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Max 20 images, 10MB each"
    )

    if uploaded_files:
        # Validate number of files
        if len(uploaded_files) > 20:
            st.error("Maximum 20 images allowed. Please reduce the number of files.")
        else:
            # Validate file sizes
            for file in uploaded_files:
                if file.size > 10 * 1024 * 1024:  # 10MB
                    st.error(f"File {file.name} exceeds 10MB limit.")
                    break
            
            # Process button
            if st.button("Extract Faces", type="primary"):
                with st.spinner("Processing images..."):
                    try:
                        # Process images and create ZIP
                        zip_data, processing_time = process_images(uploaded_files, confidence_threshold)
                        
                        if zip_data:
                            st.success("Face Extraction Complete!")
                            
                            # Download button
                            st.download_button(
                                label="Download ZIP File",
                                data=zip_data,
                                file_name="extracted_faces.zip",
                                mime="application/zip",
                                use_container_width=True
                            )
                        else:
                            st.warning("No faces were detected in the uploaded images.")
                    
                    except Exception as e:
                        st.error("An error occurred during processing.")
                        st.exception(e)
                    
                    finally:
                        # Clear uploaded files from memory
                        for file in uploaded_files:
                            file.close()

if __name__ == "__main__":
    main()