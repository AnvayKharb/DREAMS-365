"""
Flask Web Server for Background Place Detection
Serves web interface and API endpoints for image upload and analysis
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from background_place_detector import BackgroundPlaceDetector
import os
import uuid

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

# Initialize the detector globally (loaded once)
print("🚀 Initializing Background Place Detector...")
detector = BackgroundPlaceDetector(arch='resnet18')
print("✅ Detector ready!")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def detect_place():
    """
    API endpoint for place detection
    Accepts: multipart/form-data with 'file' field
    Returns: JSON with detection results
    """
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    # Check if file type is allowed
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type. Allowed: ' + ', '.join(ALLOWED_EXTENSIONS)}), 400
    
    try:
        # Create uploads folder if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Generate unique filename to avoid collisions
        file_ext = secure_filename(file.filename).rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}.{file_ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save the file
        file.save(filepath)
        
        # Validate that the file is actually a valid image
        try:
            from PIL import Image
            img = Image.open(filepath)
            img.verify()  # Verify it's a real image
        except Exception:
            try:
                os.remove(filepath)
            except:
                pass
            return jsonify({'success': False, 'error': 'Invalid image file. Please upload a real image (JPG, PNG, etc).'}), 400
        
        # Analyze the image
        predictions = detector.predict_place(filepath, top_k=5)
        
        if not predictions:
            try:
                os.remove(filepath)
            except:
                pass
            return jsonify({'success': False, 'error': 'Failed to analyze image. The model could not process this file.'}), 500
        
        # Format results
        results = {
            'success': True,
            'results': {
                'primary_place': predictions[0][1] if predictions else None,
                'primary_confidence': predictions[0][0] if predictions else 0,
                'top_places': [
                    {
                        'name': place,
                        'confidence': float(prob)
                    }
                    for prob, place in predictions
                ]
            }
        }
        
        # Clean up - delete the uploaded file
        try:
            os.remove(filepath)
        except:
            pass  # Ignore cleanup errors
        
        return jsonify(results)
        
    except Exception as e:
        print(f"❌ Error processing upload: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/demo')
def demo_info():
    """
    API endpoint for system information
    Returns: JSON with model capabilities and example categories
    """
    examples = {
        "Religious": ["abbey", "church/indoor", "church/outdoor", "mosque"],
        "Recreation": ["amusement_park", "beach", "playground", "park"],
        "Food & Dining": ["cafeteria", "restaurant", "fastfood_restaurant", "bar"],
        "Nature": ["forest/broadleaf", "forest_path", "lake/natural", "mountain"],
        "Work & Study": ["office", "library/indoor", "classroom", "laboratory_wet"],
        "Home": ["kitchen", "bedroom", "living_room", "bathroom"],
        "Culture": ["museum/indoor", "art_gallery", "theater/indoor"],
        "Transportation": ["airport_terminal", "train_station/platform", "subway_station/platform"]
    }
    
    return jsonify({
        'model_arch': detector.arch,
        'total_classes': len(detector.classes),
        'examples': examples
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎯 Background Place Detector Server")
    print("="*60)
    print("✅ Server starting...")
    print("📍 Open your browser and go to: http://localhost:3000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=3000)
