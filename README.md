# 🎯 Background Place Detector

AI-powered web application to identify background places in images using the Places365 deep learning model.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Server

```bash
python app.py
```

The server will start on **http://localhost:3000**

### 3. Use the Web Interface

1. Open your browser and go to: **http://localhost:3000**
2. Click the upload area or drag and drop an image
3. Click "Analyze Image"
4. View the detected background places with confidence scores

## 📸 What Can It Detect?

The system can identify **365 different place categories**, including:

### Religious

- Churches, mosques, temples, abbeys

### Recreation

- Parks, beaches, amusement parks, playgrounds

### Food & Dining

- Restaurants, cafes, bars, diners

### Nature

- Forests, mountains, lakes, oceans

### Work & Study

- Offices, libraries, classrooms, laboratories

### Home

- Kitchens, bedrooms, living rooms, bathrooms

### Culture

- Museums, art galleries, theaters, concert halls

### Transportation

- Airports, train stations, subway stations

And many more!

## 🔧 Technical Details

- **Backend**: Flask web server
- **Model**: ResNet18 trained on Places365 dataset
- **Frontend**: HTML/CSS/JavaScript
- **Image Processing**: PyTorch + Torchvision

## 📁 Project Structure

```
├── app.py                          # Flask web server
├── background_place_detector.py    # Place detection class
├── templates/
│   └── index.html                  # Web interface
├── uploads/                        # Temporary upload folder
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🌐 API Endpoints

### POST /api/detect

Upload an image for place detection

**Request**: multipart/form-data with 'file' field  
**Response**: JSON with top 5 predicted places and confidence scores

Example:

```bash
curl -X POST -F "file=@image.jpg" http://localhost:3000/api/detect
```

### GET /api/demo

Get system information and example place categories

**Response**: JSON with model info and place examples

Example:

```bash
curl http://localhost:3000/api/demo
```

## 📝 Notes

- Maximum file size: 16MB
- Supported formats: JPG, PNG, GIF, BMP, TIFF, WebP
- Model automatically downloads on first run (~45MB)
- Uploaded files are automatically cleaned up after processing

## 🎓 Model Information

- **Dataset**: Places365-Standard
- **Architecture**: ResNet18
- **Classes**: 365 scene categories
- **Paper**: [Places: A 10 million Image Database for Scene Recognition](http://places2.csail.mit.edu/)

## 🛑 Stopping the Server

Press `Ctrl+C` in the terminal where the server is running

## 📚 Command Line Usage

You can also use the detector directly from command line:

```bash
# Single image
python background_place_detector.py --image photo.jpg

# Batch process folder
python background_place_detector.py --folder ./my_photos/

# Different model architecture
python background_place_detector.py --image photo.jpg --arch resnet50

# Show more predictions
python background_place_detector.py --image photo.jpg --top-k 10
```

---

**Enjoy detecting background places! 🎉**
