# Face Recognition Project

This project uses a Convolutional Neural Network (CNN) to recognize faces in real-time. It's designed to collect face data from a mobile phone camera, train a CNN model on that data, and then use the trained model for real-time face recognition.

## 📋 Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Configuration](#configuration)

## ✨ Features
- Real-time face data collection from mobile camera
- CNN-based face recognition using LeNet-like architecture
- Automatic face detection using Haar Cascade classifiers
- Image preprocessing and data augmentation
- Real-time face recognition with confidence display

## 📦 Requirements

The project requires the following Python libraries (listed in `requirements.txt`):

- `opencv-python`: Computer vision tasks (video capture, face detection)
- `numpy`: Numerical operations and array handling
- `tensorflow`: Core machine learning framework
- `keras`: High-level neural network API
- `scikit-learn`: Machine learning utilities (label encoding)
- `matplotlib`: Data visualization and plotting

## 📁 Project Structure

```
face-recognition-project/
│
├── collect_data.py              # Face data collection script
├── consolidated_data.py         # Data preprocessing script
├── Model_CNN.py                 # CNN model training script
├── recognize.py                 # Real-time face recognition script
├── requirements.txt             # Python dependencies
├── haarcascade_frontalface_default.xml  # Haar cascade classifier (included)
├── README.md                    # This file
│
├── images/                      # Raw face images directory
│   └── [person_name]/          # Individual person directories
│
├── Clean data/                  # Processed data directory
│   ├── images.p                # Processed images (pickle file)
│   └── labels.p                # Corresponding labels (pickle file)
│
└── models/                      # Trained models directory
    ├── final_model.h5          # Trained CNN model
    └── le.p                    # Label encoder (pickle file)
```

## 🚀 Setup and Installation

### 1. Install Python
Make sure you have Python 3.7 or higher installed on your system:
- **Windows**: Download from [python.org](https://www.python.org/downloads/)
- **macOS**: Download from [python.org](https://www.python.org/downloads/) or use `brew install python3`
- **Linux**: Use your package manager, e.g., `sudo apt install python3 python3-pip`

Verify installation:
```bash
python --version
pip --version
```

### 2. Clone the Repository
```bash
git clone <your-repository-url>
cd face-recognition-project
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Create Required Directories
```bash
mkdir -p images "Clean data" models
```

## 📱 Usage

### Step 1: Collect Face Data
Run the data collection script to capture face images from your mobile camera:

```bash
python collect_data.py
```

**Before running:**
- Install a camera app on your phone that provides an IP stream (like "IP Webcam" for Android)
- Update the camera URL in `collect_data.py` (replace the example IP address)
- Ensure your phone and computer are on the same network

The script will:
- Connect to your phone's camera stream
- Detect and crop faces automatically
- Save 100 face images to the `images/` directory
- Prompt you to enter the person's name for labeling

### Step 2: Process the Data
Preprocess the collected images:

```bash
python consolidated_data.py
```

This script will:
- Resize images to 32x32 pixels
- Convert to grayscale
- Save processed data as pickle files in the `Clean data/` directory

### Step 3: Train the CNN Model
Train the face recognition model:

```bash
python Model_CNN.py
```

**Configuration required in `Model_CNN.py`:**
```python
# Update these paths
images_path = "Clean data/images.p"
labels_path = "Clean data/labels.p"
model_output = "models/final_model.h5"
label_encoder_output = "models/le.p"
```

The script will:
- Load preprocessed data
- Apply histogram equalization and normalization
- Train a LeNet-like CNN for 10 epochs
- Save the trained model and label encoder

### Step 4: Real-time Face Recognition
Run the recognition system:

```bash
python recognize.py
```

**Configuration required in `recognize.py`:**
```python
# Update these paths
model_path = "models/final_model.h5"
label_encoder_path = "models/le.p"
haar_cascade_path = "haarcascade_frontalface_default.xml"

# Update camera URL
url = "http://YOUR_PHONE_IP:8080/shot.jpg"  # Replace with your phone's camera URL
```

The script will:
- Load the trained model and label encoder
- Connect to your phone's camera stream
- Detect and recognize faces in real-time
- Display results with confidence scores

Press 'q' to quit the recognition system.

## ⚙️ Configuration

### Mobile Camera Setup

1. **For Android**: Install "IP Webcam" app
   - Set up the camera stream
   - Note the provided URL (usually `http://PHONE_IP:8080/shot.jpg`)
   
2. **For iPhone**: Install "EpocCam" or similar app
   - Follow app instructions for IP streaming

### File Path Configuration

Before running the training and recognition scripts, update the following paths:

#### In `Model_CNN.py`:
```python
images_path = "Clean data/images.p"       # Processed images input
labels_path = "Clean data/labels.p"       # Labels input
model_output = "models/final_model.h5"    # Trained model output
label_encoder_output = "models/le.p"      # Label encoder output
```

#### In `recognize.py`:
```python
model_path = "models/final_model.h5"                    # Trained model path
label_encoder_path = "models/le.p"                      # Label encoder path
haar_cascade_path = "haarcascade_frontalface_default.xml"  # Haar cascade path
url = "http://YOUR_PHONE_IP:8080/shot.jpg"              # Camera stream URL
```

#### In `collect_data.py`:
```python
haar_cascade_path = "haarcascade_frontalface_default.xml"  # Haar cascade path
url = "http://YOUR_PHONE_IP:8080/shot.jpg"                # Camera stream URL
output_dir = "images/"                                     # Output directory
```



**Note**: This project is designed for educational purposes. Ensure you have proper consent before collecting and using face data, and be aware of privacy regulations in your jurisdiction.
