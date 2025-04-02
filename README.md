# Image Segmentation Project

This project demonstrates an end-to-end image segmentation system using a pre-trained DeepLabV3 model. It provides a Flask backend that processes an uploaded image, performs semantic segmentation, overlays the segmentation mask on the original image, and returns the result as a Base64-encoded image. A simple frontend allows users to upload an image, view the annotated result, and download it if desired.

## Features

- **Semantic Segmentation:** Uses DeepLabV3 with a ResNet-50 backbone pre-trained on a standard dataset.
- **Overlay & Visualization:** Blends the segmentation mask with the original image.
- **Download Option:** Annotated images can be downloaded.
- **Responsive Frontend:** Drag & drop and file selection for image upload with progress feedback.
- **Docker Support:** Dockerfile provided for containerizing the backend.
- **CORS Enabled:** Allows the frontend and backend to communicate seamlessly.

## Project Structure
image_segmentation_project/
├── backend/
│   ├── app.py             # Flask backend code for image segmentation
│   ├── requirements.txt   # Python dependencies for backend
│   └── Dockerfile         # Docker configuration for containerization
└── frontend/
├── index.html         # HTML page with image upload form and display
├── css/
│   └── styles.css     # Custom CSS styling
└── js/
└── app.js         # JavaScript for handling image upload and API calls
## Installation

### Prerequisites

- Python 3.9 or higher
- Git

### Setup Locally

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/<YOUR_USERNAME>/image_segmentation_project.git
   cd image_segmentation_project
2.	**Install Backend Dependencies:**

    ```bash
    cd backend
    pip install -r requirements.txt

3.	**Run the Backend:**
    ```bash
    python app.py
The Flask server will run at http://localhost:5100.
4.	**Serve the Frontend:**
	•	Open frontend/index.html directly in your browser, or
	•	Serve it using a simple HTTP server:
    ```bash
    cd ../frontend
    python -m http.server 8000
Then navigate to http://localhost:8000.
**USAGE**
1.	Upload an Image:
On the frontend page, drag & drop an image or click to select a file.
2.	Processing:
The frontend will send the image to the backend. A progress indicator will display during processing.
3.	View Results:
The backend returns an annotated image (with segmentation overlays and bounding boxes) along with object counts.
4.	Download:
Use the download button to save the annotated image locally.

**Deployment**
**Docker**
To build and run the backend using Docker:
1.	Build the Docker Image:
    ```bash
    cd backend
    docker build -t image-segmentation-app .
2. Run the Docker Container:
    ```bash
    docker run -p 5100:5100 image-segmentation-app
License

This project is open source and available under the MIT License.
---

### .gitignore

```plaintext
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Virtual environments
env/
venv/

# Distribution / packaging
build/
dist/
*.egg-info/

# OS-specific files
.DS_Store
Thumbs.db

# Logs
*.log

# Editor directories and files
.vscode/
.idea/

# Python virtualenv
*.env

# Docker files (if you wish to ignore local Docker artifacts)
docker-compose.override.yml