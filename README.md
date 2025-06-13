# FaceXtract | ASAI

A streamlined web application for extracting individual face images from classroom or group photos.

## Features

- Upload multiple images (JPG, JPEG, PNG)
- Advanced face detection using MTCNN
- Adjustable confidence threshold for detection accuracy
- Batch processing with progress tracking
- Download extracted faces in ZIP format
- Mobile-friendly responsive design

## Technical Stack

- Streamlit
- TensorFlow
- MTCNN
- OpenCV
- Python 3.x

## Local Development

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run streamlit_app.py
   ```

## Deployment

This application is deployed on Streamlit Cloud. Visit [https://facextract.streamlit.app](https://facextract.streamlit.app) to use the application.

## License

MIT License 